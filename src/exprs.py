import polars as pl
import pyarrow as pa
import re
from .core import AttrExpr, Chunks

from typing import Literal, List, Tuple, Union

__all__ = [
    "AtomData",
    "RegexCount",
    "ChunkOverlap",
    "SimpleStringify",
    "ChunkDelimitedStringify",
]


class AtomData(AttrExpr):
    """
    Get atom data for each chunk. Results in a list of values for each
    chunk (one per atom in the chunk).

    Parameters
    ----------
    attr
        Name of the *atom* attribute to include.

    Warnings
    --------
    Order of results within the list is not guaranteed.
    """

    def __init__(self, attr: str):
        self.attr = attr

    def __call__(self, chunks: Chunks) -> pa.Array:
        return (
            # Start with `chunks` to ensure order and keep out unrelated atoms
            pl.from_arrow(chunks.chunks)
            .select(chunk="id")  # aliasing to "chunk"
            # Find the atoms for these chunks
            .join(pl.from_arrow(chunks.chunk_atoms), left_on="chunk", right_on="chunk")
            # Join with individual atom data
            .join(pl.from_arrow(chunks.corpus.atoms), left_on="atom", right_on="id")
            .group_by("chunk", maintain_order=True)
            .agg(self.attr)
            .get_column(self.attr)
            .to_arrow()
        )


class RegexCount(AttrExpr):
    """
    Count how many matches to a regular expression exist in a given chunk, and add a
    regex match count column to the chunk attributes.

    Parameters
    ----------
    stringifier
        An `AttrExpression` that returns one string per chunk; its output
        is used as the input to the regex engine.
    pattern
        Regex pattern to match against.
    flags
        Regular expression flag(s). To include multiple flags you can combine them
        using the bitwise OR operator "|".
    Examples
    --------
    Normal regex:
    >>> RegexCount(SimpleStringifier(), r"DOB")

    Case-insensitive, via inline flag:
    >>> RegexCount(SimpleStringifier(), r"(?i)DOB")

    Case-insensitive via flags parameter:
    >>> RegexCount(SimpleStringifier(), r"DOB", flags=re.IGNORECASE | re.MULTILINE)

    Numeric flags:
    >>> RegexCount(chunks, flags=2 | 8)  # 2 = re.IGNORCARE, 8 = re.MULTILINE

    Sum of the flag integers will apply both flags:
    >>> RegexCount(chunks, flags=10)  # 2 + 8 == 10
    """

    def __init__(
        self, stringifier: AttrExpr, pattern: str, flags: Union[int, re.RegexFlag] = 0
    ):
        self.stringifier = stringifier
        self.pattern = pattern
        self.flags = flags

    def __call__(self, chunks: Chunks) -> pa.Array:
        # (The stringifier is responsible for returning its strings
        # in the correct order for the input chunks.)

        strings = self.stringifier(chunks).to_pylist()

        # There are probably faster ways to regex over a list (e.g. Polars can)
        # but the built-in `re` module is definitely more familiar to people.
        match_counts = [
            len(re.findall(self.pattern, text, self.flags)) for text in strings
        ]

        return pa.array(match_counts)


class ChunkOverlap(AttrExpr):
    """
    Calcuate whether (or how much) each chunk overlaps with another kind of chunk.

    Parameters
    ----------
    chunk_b
        ChunkExpr for the chunks to check overlap against.
    agg : {"bool", "count", "frac"}
        How the overlap value is aggregated and reported.
        * `bool`
            Returns a boolean value that indicates whether there was an overlap or not.
        * `count`
            Return the number of atoms in the overlap.
        * `frac`
            Return the proportion of atoms in `chunk_a` that are also in `chunk_b`.
    """

    def __init__(
        self,
        chunk_b,
        agg: Literal["bool", "count", "frac"],
    ):
        self.chunk_b = chunk_b
        self.agg = agg

        self.agg_exprs = {
            "bool": pl.col("chunk_b").is_not_null().any(),
            "count": pl.col("chunk_b").is_not_null().sum(),
            "frac": pl.col("chunk_b").is_not_null().mean(),
        }

        if self.agg not in self.agg_exprs:
            raise ValueError(f"`agg` kwarg should be one of: {self.agg_exprs.keys()}")

    def __call__(self, chunks: Chunks) -> pa.Array:
        """
        Check the `self.chunk_a` table of `corpus` against the `self.chunk_b` for
        overlapping atoms, and put the resulting overlap values in the `self.name` column
        of the `self.chunk_a` table.
        """
        corpus = chunks.corpus

        overlaps = (
            # Join chunk_a and chunk_b
            pl.from_arrow(chunks.chunk_atoms)
            .join(
                pl.from_arrow(corpus.chunk(self.chunk_b).chunk_atoms).select(
                    pl.col("chunk").alias("chunk_b"), "atom"
                ),
                on="atom",
                how="left",
            )
            # .join(
            #     pl.from_arrow(corpus["atom"]),
            #     on="atom",
            # )
            # Aggregate shared atoms along chunk_a. Need to ensure order isn't messed up.
            .group_by(pl.col("chunk"), maintain_order=True)
            # Use selected aggregation method
            .agg(overlap=self.agg_exprs[self.agg])
            .get_column("overlap")
        )

        return overlaps.to_arrow()


class SimpleStringify(AttrExpr):
    """
    Creates a list of strings from a collection of retrieved chunks.

    Input to the Stringifier is the output from a Filter.

    Parameters
    ----------
    delimiter
        String value to use when joining atoms of chunks. Defaults to " ".
    """

    def __init__(self, delimiter: str = " "):
        self.delimiter = delimiter

    def __call__(self, chunks: Chunks) -> pa.Array:
        # Get the original atom data.
        atom = chunks.corpus.atoms

        if "text" not in atom.schema.names:
            raise ValueError(
                "Stringifying requires the corpus' atoms to have a `text` column, but none was found."
            )
        if "ordinal" not in atom.schema.names:
            raise ValueError(
                "Stringifying requires the corpus' atoms to have an `ordinal` column, but none was found."
            )

        # Figure out which atoms are in these chunks, and get all relevant atom attrs.
        input_chunks = pl.from_arrow(chunks.chunks).select("id").rename({"id": "chunk"})
        enriched = input_chunks.join(
            pl.from_arrow(chunks.chunk_atoms),
            on="chunk",
        ).join(pl.from_arrow(atom).rename({"id": "atom"}), on="atom")

        res = enriched.group_by(["chunk"]).agg(
            pl.col("text").sort_by("ordinal").str.join(self.delimiter)
        )

        # Re-join with inputs to ensure correct ordering.
        res = input_chunks.join(res, on="chunk", how="left").get_column("text")
        # (`.get_column` makes a pl.Series)

        return res.to_arrow()  # Makes a pa.Array


class ChunkDelimitedStringify(AttrExpr):
    """
    Converts chunks into strings, with customizable chunk-based delimiters:
    if transitions between the provided chunks are encountered within the
    retrieved data, the designated delimiter will be used to join the chunks.

    Parameters
    ----------
    chunk_delimiters
        List of `(chunk, delimiter)` pairs, both strings. Delimiters are
        applied in order, so if you have a new `page` *and* a new `block`, with
        different delimiters for both, whichever chunk type occurs first
        in the `chunk_delimiters` list will have its delimiter used, and any
        other potentially-applicable delimiters will be disregarded.
    atom_delimiter
        Which string value to join individual atoms on. Defaults to `" "`.
    """

    def __init__(
        self, chunk_delimiters: List[Tuple[str, str]], *, atom_delimiter: str = " "
    ):
        self.chunk_delimiters = chunk_delimiters
        self.atom_delimiter = atom_delimiter

    def __call__(self, chunks: Chunks) -> pa.Array:
        # Get the original corpus/atom data.
        corpus = chunks.corpus
        atoms = corpus.atoms

        # Figure out which atoms are in these chunks, and get all relevant atom attrs.
        # Results *must* be ordered according to the input chunks, hence the `left`
        # joins and `maintain_order`s throughout (these maintain order in Polars)
        enriched = (
            pl.from_arrow(chunks.chunks)
            .select("id")
            .rename({"id": "chunk"})
            .join(pl.from_arrow(chunks.chunk_atoms), on="chunk", how="left")
            .join(pl.from_arrow(atoms).rename({"id": "atom"}), on="atom", how="left")
        )

        for chunk, _ in self.chunk_delimiters:
            # Make sure there are not duplicate columns
            if chunk in enriched.columns:
                enriched = enriched.drop(chunk)

            # Get chunk IDs for applicable chunks, so we can identify diffs/changes.
            # (This involves reaching up into the full corpus for external chunk info.)
            enriched = enriched.join(
                pl.from_arrow(corpus.chunk(chunk).chunk_atoms).select(
                    pl.col("chunk").alias(chunk), "atom"
                ),
                on="atom",
                how="left",
            )

        # Dynamically chain together an expression for conditional spacers
        expr = pl
        for chunk, delimiter in self.chunk_delimiters:
            # Check if current chunk ID != next chunk ID
            expr = expr.when(pl.col(chunk).ne(pl.col(chunk).shift(-1))).then(
                pl.lit(delimiter)
            )

        # Set final row spacer to empty string to avoid trailing spaces
        expr = expr.when(pl.int_range(0, pl.len()).eq(pl.len() - 1)).then(pl.lit(""))

        # Add default delimiter (between atoms) to everything else.
        expr = expr.otherwise(pl.lit(self.atom_delimiter))

        # Materialize the chunks.
        results = (
            enriched.group_by(["chunk"], maintain_order=True)
            .agg(
                text=(
                    pl.concat_str(pl.col("text"), expr).sort_by("ordinal").str.join("")
                )
            )
            .get_column("text")  # Result is a pl.Series
        )

        return results.to_arrow()  # Makes a pa.Array
