from retrievall.core import Chunks, Corpus
import polars as pl
import pyarrow as pa

__all__ = [
    "corpus_from_tesseract_table",
]


def corpus_from_tesseract_table(ocr_table: pa.Table, document_id) -> Corpus:
    """
    Convert a (PyArrow) Tesseract OCR table to a Corpus, with OCR tokens as its atoms.

    An OCR token has, minimally:
    * Text content.
    * Ordinal information (to determine reading order).
    * Spatial information (e.g. bounding box info). May be relative to a page
      (e.g a fraction), or may be absolute (e.g. pixel amounts). In Tesseract
      formats, absolute units are used.

    This function provides creates  a corpus with the following data:
    * Atoms, with `text`, `ordinal`, bounding box, and `confidence` data.
    * `page`, `block`, `paragraph`, and `line` chunks, each with `ordinal`, bounding
      box, and parent data (where applicable)
    * A `document` chunk that encompases all tokens from the Tesseract object (useful
      in corpora with more than one document).
    """
    ocr_table = pl.from_arrow(ocr_table)

    page, block, par, line, atom = (
        ocr_table
        # Ensure correct order, just to be safe
        .sort("page_num", "block_num", "par_num", "line_num", "word_num")
        .with_columns(
            document=pl.lit(document_id),
            # Make all ordinal values cumulative along the whole document, instead of
            # resetting for each container type.
            word_num=pl.when(pl.col("level") >= 5).then(
                pl.col("level").eq(5).cum_sum()
            ),
            line_num=pl.when(pl.col("level") >= 4).then(
                pl.col("level").eq(4).cum_sum()
            ),
            par_num=pl.when(pl.col("level") >= 3).then(pl.col("level").eq(3).cum_sum()),
            block_num=pl.when(pl.col("level") >= 2).then(
                pl.col("level").eq(2).cum_sum()
            ),
            page_num=pl.when(pl.col("level") >= 1).then(
                pl.col("level").eq(1).cum_sum()
            ),
            # Set nulls for levels that don't have valid conf/text values
            conf=pl.when(pl.col("level") == 5).then(pl.col("conf")),
            text=pl.when(pl.col("level") == 5).then(pl.col("text")),
            # Store tesseract coordinates at atom level, for potential downstream use
            tesseract_coord=pl.struct(
                "page_num",
                "block_num",
                "par_num",
                "line_num",
                "word_num",
            ),
        )
        .with_columns(
            # Hash based on re-aligned ordinal values
            id=pl.struct(
                "document",
                "page_num",
                "block_num",
                "par_num",
                "line_num",
                "word_num",
            ).hash()
        )
        # We know the output order will be page/block/etc. because of sorting.
        .partition_by("level", maintain_order=True, include_key=False)
    )

    # Clean up tables.
    # NOTE: Everything above atoms is storing relationship data in the objects
    # (e.g. parent object IDs, like a block's page ID). That may not be the
    # best approach in the long run.
    document = pl.DataFrame({"id": document_id})

    page = page.select(
        pl.col("document"),
        pl.col("width"),
        pl.col("height"),
        ordinal=pl.col("page_num"),
        id=pl.col("id"),
    )

    block = block.join(
        page.rename({"id": "page"}), left_on="page_num", right_on="ordinal"
    ).select(
        pl.col("page"),
        pl.col("left"),
        pl.col("top"),
        pl.col("width"),
        pl.col("height"),
        ordinal=pl.col("block_num"),
        id=pl.col("id"),
    )

    par = par.join(
        block.rename({"id": "block"}), left_on="block_num", right_on="ordinal"
    ).select(
        pl.col("block"),
        pl.col("left"),
        pl.col("top"),
        pl.col("width"),
        pl.col("height"),
        ordinal=pl.col("par_num"),
        id=pl.col("id"),
    )

    line = line.join(
        par.rename({"id": "paragraph"}), left_on="par_num", right_on="ordinal"
    ).select(
        pl.col("paragraph"),
        pl.col("left"),
        pl.col("top"),
        pl.col("width"),
        pl.col("height"),
        ordinal=pl.col("line_num"),
        id=pl.col("id"),
    )

    # Record atom relations
    document_atom = (
        document.select(pl.col("id").alias("chunk"))
        .join(atom, left_on=["chunk"], right_on=["document"])
        .rename({"id": "atom"})
        .select("chunk", "atom")
    )

    page_atom = (
        page.select(pl.col("id").alias("chunk"), "ordinal")
        .join(atom, left_on=["ordinal"], right_on=["page_num"])
        .rename({"id": "atom"})
        .select("chunk", "atom")
    )

    block_atom = (
        block.select(pl.col("id").alias("chunk"), "ordinal")
        .join(atom, left_on=["ordinal"], right_on=["block_num"])
        .rename({"id": "atom"})
        .select("chunk", "atom")
    )

    paragraph_atom = (
        par.select(pl.col("id").alias("chunk"), "ordinal")
        .join(atom, left_on=["ordinal"], right_on=["par_num"])
        .rename({"id": "atom"})
        .select("chunk", "atom")
    )

    line_atom = (
        line.select(pl.col("id").alias("chunk"), "ordinal")
        .join(atom, left_on=["ordinal"], right_on=["line_num"])
        .rename({"id": "atom"})
        .select("chunk", "atom")
    )

    # Clean up atoms
    atom = atom.select(
        pl.col("left"),
        pl.col("top"),
        pl.col("width"),
        pl.col("height"),
        pl.col("text"),
        confidence=pl.col("conf"),
        ordinal=pl.col("word_num"),
        id=pl.col("id"),
        tesseract_coord=pl.col("tesseract_coord"),
        document=pl.col("document"),
    )

    # Set up corpus (must use PyArrow tables!)
    corpus = Corpus(atoms=atom.to_arrow())

    # Add chunks
    corpus.set_chunk(
        "document", Chunks(corpus, document.to_arrow(), document_atom.to_arrow())
    )
    corpus.set_chunk("page", Chunks(corpus, page.to_arrow(), page_atom.to_arrow()))
    corpus.set_chunk("block", Chunks(corpus, block.to_arrow(), block_atom.to_arrow()))
    corpus.set_chunk(
        "paragraph", Chunks(corpus, par.to_arrow(), paragraph_atom.to_arrow())
    )
    corpus.set_chunk("line", Chunks(corpus, line.to_arrow(), line_atom.to_arrow()))

    return corpus
