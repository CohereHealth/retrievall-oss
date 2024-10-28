import re
from .core import ChunkExpr, Chunks, Corpus
import polars as pl


__all__ = [
    "FixedSizeChunk",
    "RegexMatchChunk",
]


class FixedSizeChunk(ChunkExpr):
    """
    Chunks documents into fixed-size pieces, with each chunk having the same
    number of elements. These chunks can be disjoint, or they can have a
    specified amount of overlap.

    Parameters
    ----------
    constrain_to
        **Required.** Which chunks should bound the fixed-size chunking process. For
        example, if you want 100-word chunks, do you want those windows to be bounded
        within `document`s? `page`s? Some other kind of existing chunk?
    size
        **Required.** Number of atoms per chunk.
    offset
        How far the start of one chunk is from the end of another. Can be negative
        to cause overlap, or positive to cause gaps.
    closed : {"left", "right", "both", "none"}
        Whether thr chunks are open (exclusive) or closed (inclusive) on their edges. See
        https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by_dynamic.html
        for more details.

    Returns
    -------
    Chunks
    """

    def __init__(
        self,
        constrain_to: str,
        size: int,
        offset: int = 0,
        *,
        closed: str = "left",
    ):
        self.size = size
        self.offset = offset
        self.constraint = constrain_to
        self.closed = closed

    def __call__(self, corpus: Corpus) -> Chunks:
        if "ordinal" not in corpus.atoms.schema.names:
            raise KeyError(
                "Fixed-size chunking requires the atoms to have `ordinal` column, but none was found."
            )

        # We need chunk-per-atom data so that chunks can be properly ordered.
        joined = (
            pl.from_arrow(corpus.chunk(self.constraint).chunk_atoms)
            .rename({"chunk": "constraint"})
            .join(pl.from_arrow(corpus.atoms), left_on="atom", right_on="id")
        )

        chunks = (
            joined.sort(["constraint", "ordinal"])
            .group_by_dynamic(
                pl.col("ordinal").cast(pl.Int64),
                every=f"{self.size + self.offset}i",
                period=f"{self.size}i",
                closed=self.closed,
                group_by="constraint",
                start_by="datapoint",
            )
            .agg(pl.col("atom"))
            .with_columns(
                pl.struct(pl.col("constraint", "ordinal")).hash().alias("id"),
            )
        )

        return Chunks(
            corpus=corpus,
            chunks=chunks.select("id").to_arrow(),
            chunk_atoms=chunks.select(pl.col("id").alias("chunk"), "atom")
            .explode("atom")
            .to_arrow(),
        )


class RegexMatchChunk(ChunkExpr):
    """
    Make chunks from regex matches.

    Parameters
    ----------
    constrain_to
        Which chunks should bound the regex matches. For example, do you want your matches
        to be bound within `document`s? `page`s? Some other kind of existing chunk?
        Defaults to `document`.
    pattern
        Pattern to match

    Returns
    -------
    Chunks
    """

    def __init__(
        self,
        constrain_to: str,
        pattern: str,
    ):
        self.constraint = constrain_to
        self.pattern = pattern

    def __call__(self, corpus: Corpus) -> Chunks:
        if "ordinal" not in corpus.atoms.schema.names:
            raise KeyError(
                "Regex-match chunking requires the atoms to have `ordinal` column, but none was found."
            )

        # We'll be joining text with spaces. This might be something we want to
        # be modifyable in the future...
        SPACER = " "

        # We need chunk-per-atom data so that chunks can be properly ordered.
        joined = (
            pl.from_arrow(corpus.chunk(self.constraint).chunk_atoms)
            .rename({"chunk": "constraint"})
            .join(pl.from_arrow(corpus.atoms), left_on="atom", right_on="id")
        )
        indices = joined.sort(["constraint", "ordinal"]).select(
            pl.col("constraint"),
            pl.col("atom"),
            pl.col("text")
            .str.len_chars()
            .add(pl.lit(SPACER).str.len_chars())
            .cum_sum()
            .shift(1, fill_value=0)
            .over(["constraint"])
            .alias("start_index"),
        )

        chunks = (
            joined.sort(["constraint", "ordinal"])
            .group_by(["constraint"])
            .agg(
                pl.col("text")
                .str.join(SPACER)
                .map_elements(
                    lambda x: [
                        dict(zip(["start", "end"], m.span()))
                        for m in re.finditer(self.pattern, x)
                    ],
                    return_dtype=pl.List(
                        pl.Struct({"start": pl.Int64, "end": pl.Int64})
                    ),
                )
                .alias("match")
            )
            .explode("match")
            # Join/filter where start_index is between `match` start/end values
            .join(indices, on="constraint")
            .filter(
                pl.col("start_index").is_between(
                    pl.col("match").struct.field("start"),
                    pl.col("match").struct.field("end"),
                )
            )
            # Our chunks are anywhere with unique `constraint`s and `match`es. Here
            # we'll "unnest" the match struct by accessing the start and end fields.
            .group_by(
                pl.struct(
                    "constraint",
                    pl.col("match").struct.field("start"),
                    pl.col("match").struct.field("end"),
                )
                .hash()  # Hash to get unique chunk IDs.
                .alias("id"),
            )
            .agg("atom")
        )

        return Chunks(
            corpus=corpus,
            chunks=chunks.select("id").to_arrow(),
            chunk_atoms=chunks.select(pl.col("id").alias("chunk"), "atom")
            .explode("atom")
            .to_arrow(),
        )
