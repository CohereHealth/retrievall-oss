import pyarrow as pa
from retrievall import Chunks, Corpus
from retrievall.ocr import corpus_from_tesseract_table


class TestOCRCorpus:
    def test_from_tesseract(self, tesseract_table):
        corpus = corpus_from_tesseract_table(tesseract_table, document_id="abc123")
        assert isinstance(corpus, Corpus)

        # Verify tables made it to the right places.
        assert isinstance(corpus.atoms, pa.Table)
        assert isinstance(corpus.chunk("document"), Chunks)
        assert isinstance(corpus.chunk("page"), Chunks)
        assert isinstance(corpus.chunk("block"), Chunks)
        assert isinstance(corpus.chunk("paragraph"), Chunks)
        assert isinstance(corpus.chunk("line"), Chunks)

        # Verify tables have the right number of tokens, based on the input.
        assert len(corpus.atoms) == 18
        assert len(corpus.chunk("document")) == 1
        assert len(corpus.chunk("page")) == 2
        assert len(corpus.chunk("block")) == 2
        assert len(corpus.chunk("paragraph")) == 4
        assert len(corpus.chunk("line")) == 8

    def test_merge(self, tesseract_table):
        corpus1 = corpus_from_tesseract_table(tesseract_table, document_id="abc123")
        corpus2 = corpus_from_tesseract_table(tesseract_table, document_id="def456")

        merged = Corpus.merge([corpus1, corpus2])

        assert isinstance(merged, Corpus)
        # Check that all documents are represented.
        assert set(merged.chunk("document").chunks["id"].to_pylist()) == set(
            ["abc123", "def456"]
        )

        # Verify tables have the right number of tokens, based on the input.
        assert len(merged.atoms) == 36
        assert len(merged.chunk("page")) == 4
        assert len(merged.chunk("block")) == 4
        assert len(merged.chunk("paragraph")) == 8
        assert len(merged.chunk("line")) == 16
