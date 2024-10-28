from retrievall import Chunks, Corpus
from retrievall.ocr import corpus_from_tesseract_table
from retrievall.chunkers import (
    FixedSizeChunk,
    RegexMatchChunk,
)
from retrievall.exprs import SimpleStringify


class TestFixedSizeChunker:
    def test_chunker(self, ocr_corpus):
        corpus = ocr_corpus
        chunker = FixedSizeChunk(constrain_to="document", size=8, offset=-2)

        # Chunk corpus
        chunks = chunker(corpus)

        assert isinstance(chunks, Chunks)

        # 18 tokens in the doc, size 8 window with 6 offset == 3 chunks
        assert len(chunks) == 3

    def test_constrained_chunks(self, ocr_corpus):
        corpus = ocr_corpus
        # Lines are 2-3 words, so `3` for size and `-1` for offset gets us
        # an extra chunk for each 3-token line.
        chunker = FixedSizeChunk(constrain_to="line", size=3, offset=-1)

        # Chunk corpus
        chunks = chunker(corpus)

        assert isinstance(chunks, Chunks)

        # 6 2-word lines plus 2 3-word lines (chunked into 2 chunks each) == 10 chunks
        assert len(chunks) == 10


class TestRegexMatchChunker:
    def test_chunker(self, ocr_corpus):
        corpus = ocr_corpus
        chunker = RegexMatchChunk(constrain_to="document", pattern="Over the <lazy>")

        # Chunk corpus
        chunks = chunker(corpus)

        assert isinstance(chunks, Chunks)

        # One match should exist
        assert len(chunks) == 1

        # Chunk should materialize to "Over the <lazy>"
        res = chunks.select(text=SimpleStringify(delimiter=" "))
        assert res["text"].to_pylist() == ["Over the <lazy>"]

    # Test multiple docs...
    def test_chunker_multidoc(self, tesseract_table):
        # Create two corpora from a repeated tesseract table.
        corp1 = corpus_from_tesseract_table(tesseract_table, document_id="abc123")
        corp2 = corpus_from_tesseract_table(tesseract_table, document_id="edf456")

        # Merge corpora
        corpus = Corpus.merge([corp1, corp2])

        # Set up chunker
        chunker = RegexMatchChunk(constrain_to="document", pattern="Over the <lazy>")

        # Chunk corpus
        chunks = chunker(corpus)

        assert isinstance(chunks, Chunks)

        # Two matches should exist (one per doc)
        assert len(chunks) == 2

        # # Chunks should materialize to "Over the <lazy>" x 2
        res = chunks.select(text=SimpleStringify(delimiter=" "))
        assert res["text"].to_pylist() == ["Over the <lazy>", "Over the <lazy>"]

    def test_chunker_constrained(self, ocr_corpus):
        corpus = ocr_corpus
        chunker = RegexMatchChunk(constrain_to="line", pattern="Over the <lazy>")

        # Chunk corpus
        chunks = chunker(corpus)

        assert isinstance(chunks, Chunks)

        # *NO* match should exist, because the full string can't be found.
        assert len(chunks) == 0
