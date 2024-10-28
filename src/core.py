from __future__ import annotations  # For circular/"forward reference" annotations
from abc import ABC, abstractmethod
import pyarrow as pa

from typing import Collection

__all__ = [
    "Corpus",
    "Chunks",
    "ChunkExpr",
    "AttrExpr",
    "ChunkFilter",
]


class ChunkExpr(ABC):
    """
    `ChunkExpr`s are expressions for *chunking* documents in a `Corpus`.

    A `ChunkExpr` is a callable object, whose `__call__` method takes a `Corpus` and
    returns a `Chunks` object.

    `ChunkExpr`s are typically used within a `corpus.chunk()` call to create new
    chunks.
    """

    @abstractmethod
    def __call__(self, corpus: Corpus) -> Chunks:
        pass


class AttrExpr(ABC):
    """
    `AttrExpr`s ("attribute expressions") are expressions for adding attribute data
    to chunks—things like relevance scores, metadata, materialized representations of
    a chunk's contents, etc.

    An `AttrExpr` is a callable object, whose `__call__` method takes a `Chunks` object
    and returns a PyArrow `Array`.

    `AttrExpr`s are typically used within a `Chunks.enrich()` or `Chunks.select()` call
    to either add or materialize data about chunks.
    """

    @abstractmethod
    def __call__(self, chunks: Chunks) -> pa.Array:
        pass


class ChunkFilter(ABC):
    """
    `ChunkFilter`s take a `Chunks` object and filter out some of the chunks, based on
    some criteria.

    A `ChunkFilter` is a callable object, whose `__call__` method takes a `Chunks`
    object and returns that same `Chunks` object with a modified `Chunks.chunks` table.

    `AttrExpr`s are typically used within a `Chunks.filter()` call.
    """

    def __call__(self, chunks: Chunks) -> Chunks:
        pass


class Corpus:
    """
    A container for document(s) being analyzed, broken down into their
    constituent parts (e.g. tokens, bytes, OCR, etc.).

    Internally, this is a collection of "atoms" (i.e. the minimal "constituent parts"
    just mentioned) and collections of "chunks", which are views/collections those
    atoms. These are all represented within the corpus as various PyArrow tables, which
    are compatible with most Python DataFrame/tabular manipulation libraries.

    Retrieving from a corpus involves selecting (or creating) chunks, filtering
    out less-relevant ones based on attributes or the contents of the chunks, and
    aggregating/materializing those contents or other metadata as desired.
    """

    # Beyond atoms and chunks, it might be useful to allow additional tables that
    # store other metadata or information about the document, corpus, etc.—stuff like
    # inter-chunk relationships. Possible examples: page details, element hierarchy
    # data, or calculated/derived date info.
    def __init__(self, atoms: pa.Table):
        # We can't instantiate a Corpus with pre-existing chunks, because a `Chunks`
        # object requires a reference to an existing corpus. So chunks have to be
        # added after instantiation.
        self.chunks = {}
        self.atoms = atoms

    def chunk(self, chunk_expr: ChunkExpr | str) -> Chunks:
        """
        Access an existing collection of chunks in the corpus, or ephemerally
        create a new collection of chunks according the given chunk expression.

        Parameters
        ----------
        chunk_expr
            Either the name of an existing chunk to access (`str`) or a chunk
            expression from which a new chunk will be created.

        Returns
        -------
        Chunks
        """
        # Return an existing chunk
        if isinstance(chunk_expr, str):
            return self.chunks[chunk_expr]

        # Call the expression on the corpus to create a new chunk.
        return chunk_expr(self)

    def set_chunk(self, name: str, chunks: Chunks):
        """
        Persist a collection of chunks (i.e. a `Chunks` object) to the corpus,
        accessible via `name`.

        Does not return anything; not intended for use in a chained-together pipeline.

        Parameters
        ----------
        name
            Name to assign the chunks to.
        chunks
            `Chunks` object to assign.
        """
        if chunks.corpus is not self:
            raise ValueError(
                "Input chunks `corpus` must be the same exact object as the corpus it's being set on. "
            )
        self.chunks[name] = chunks

    @classmethod
    def merge(cls, corpora: Collection[Corpus]) -> Corpus:
        """
        Combine multiple `Corpus` objects into a single corpus, by merging
        their chunks by name.

        Warnings
        --------
        This function does not guarantee that the input corpora are compatible.
        Be aware that:
        * Trying to merge fundamentally different kinds of `Corpus` may cause errors.
        * If ID values (e.g. atom IDs) haven't been carefully defined to avoid
        collisions, you may end up in a state where you can't tell which item
        is which and break your ability to correctly retrieve things.
        * This funciton does nothing to differentaite the input corpora, so if you
        don't do so manually beforehand, it may be impossible to tell what came from
        where. This can be avoided by creating a chunk that encompases all items in
        each individual corpus before merging (e.g. creating a `document` chunk with
        a distinct ID, which contains all atoms in the input corpus.)
        """
        keys = set(k for corp in corpora for k in corp.chunks.keys())

        # Collect all the atoms to form the basis of a merged corpus.
        merged_corpus = cls(pa.concat_tables([corp.atoms for corp in corpora]))

        # Each distinct key should end as its own chunk in `self.chunks`,
        # formed from the contents of all the input corpus chunks with that name.
        for k in keys:
            k_chunks = [corp.chunk(k) for corp in corpora]

            # Set chunks value `k` to a `Chunks` object derived from
            # combining chunks across all corpora.
            merged_corpus.chunks[k] = Chunks(
                corpus=merged_corpus,
                chunks=pa.concat_tables([c.chunks for c in k_chunks]),
                chunk_atoms=pa.concat_tables([c.chunk_atoms for c in k_chunks]),
            )

        return merged_corpus


class Chunks:
    """
    A collection of "chunks" of a document. Each chunk consists of a collection of
    atomic elements—characters, tokens, etc.—and can have any number of additional
    descriptive attributes/metadata (e.g. text content, location, sentiment, etc.)


    Parameters
    ----------
    corpus
        Parent corpus (which may include this `Chunks` object itself)
    chunks
        Table of individual chunks and their attributes. Requires at least a `chunk`
        column, containing unique IDs for each chunk. All other columns are optional,
        additional attributes for each chunk.
    chunk_atoms
        A normalized chunk/atom relationship table, where the atoms that belong to each
        chunk are listed.
    """

    def __init__(self, corpus: Corpus, chunks: pa.Table, chunk_atoms: pa.Table):
        self.corpus = corpus
        self.chunks = chunks
        self.chunk_atoms = chunk_atoms

    def __len__(self) -> int:
        "Get the number of chunks."
        return len(self.chunks)

    def enrich(self, **exprs: AttrExpr | str) -> Chunks:
        """
        Add new attributes (i.e. columns) to the chunks. Somewhat comparable to
        Polars' `DataFrame.with_columns()` or Ibis' `Table.mutate()`.

        Parameters
        ----------
        exprs
            Any number of attribute expressions. The parameter name supplied will be
            the name of the column.
            For non-identifier-friendly column names (e.g. `"my.column"`), use dict unpacking.
        """
        enriched = self.chunks
        for name, expr in exprs.items():
            enriched = enriched.append_column(name, expr(self))

        return Chunks(corpus=self.corpus, chunks=enriched, chunk_atoms=self.chunk_atoms)

    def filter(self, *filters) -> Chunks:
        """
        Filter chunks. Multiple filter expressions can be used; they will be applied
        *sequentially* in the order they are provided in.

        Parameters
        ----------
        filters
            Any number of filter expressions.
        """
        res = self

        for f in filters:
            res = f(res)

        return res

    def select(self, *attrs: str, **exprs: AttrExpr | str) -> pa.Table:
        """
        Materialize the given attributes or expressions into a PyArrow table.

        Parameters
        ----------
        attrs
            Any number of *existing* attribute names.
        exprs
            Any number of attribute expressions. The parameter name supplied will be
            the name of the column. An existing attribute can be aliased with a new
            name by passing the attribute name as an argument with a new name as the
            parameter.
        """

        selected = self.chunks.select(attrs)

        for name, expr in exprs.items():
            values = self.chunks.column(expr) if isinstance(expr, str) else expr(self)
            selected = selected.append_column(name, values)

        return selected
