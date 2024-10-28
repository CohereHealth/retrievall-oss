import pytest
import pyarrow as pa
import re
from retrievall.chunkers import RegexMatchChunk
from retrievall.exprs import (
    ChunkDelimitedStringify,
    SimpleStringify,
    RegexCount,
    ChunkOverlap,
    AtomData,
)
from retrievall.filters import Threshold
from retrievall.sparsetext import (
    Tfidf,
)


class TestAPI:
    def test_api(self, ocr_corpus):
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal",  # Check positional args
            o="ordinal",  # Check "renaming" via kwargs
            t=SimpleStringify(delimiter=""),  # Check named Expr-based kwargs
        )

        assert len(res) == 2

        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "o": 1, "t": "The(quick)[brown]foxjumps!Overthe<lazy>dog"},
            {"ordinal": 2, "o": 2, "t": "The~groovyminute!dogboundsUPONthesleepyfox"},
        ]


class TestTfidf:
    def test_expr(self, ocr_corpus):
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", tfidf=Tfidf(SimpleStringify(), query="the")
        )

        # 2 page chunks -> 2 results
        assert len(res) == 2

        # Both pages have the same amount of `the`s, with a tfidf of ≈0.5
        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "tfidf": pytest.approx(0.501, abs=1e-3)},
            {"ordinal": 2, "tfidf": pytest.approx(0.501, abs=1e-3)},
        ]


class TestAtomData:
    def test_expr(self, ocr_corpus):
        corpus = ocr_corpus

        res = (
            corpus.chunk("line")
            # Ensuring filtering doesn't mess anything up
            .filter(Threshold("ordinal", "<=", 2))
            .select(
                "ordinal",
                tesseract=AtomData("text"),
                document=AtomData("document"),
            )
            .sort_by("ordinal")  # Sorting to make assert checks consistent
        )

        # 2 line chunks -> 2 results
        assert len(res) == 2

        # Verify we're looking at the right lines in the right order
        assert res["ordinal"].to_pylist() == [1, 2]

        # Number of tokens on each line. (`list_value_length` gets the lengths of each list)
        assert pa.compute.list_value_length(res["tesseract"]).to_pylist() == [2, 3]

        # Check that document IDs are correctly associated with atoms. It's just 2 or 3
        # repeats of `abc123`
        assert res["document"].to_pylist() == [["abc123"] * 2, ["abc123"] * 3]


class TestRegexCount:
    def test_expr(self, ocr_corpus):
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal", rcount=RegexCount(SimpleStringify(), pattern=r"o")
        )

        # 2 page chunks -> 2 results
        assert len(res) == 2

        # One page has 3 `o`s, the other has 5
        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "rcount": 3},
            {"ordinal": 2, "rcount": 5},
        ]

        # Test inline flag syntax (`(?aiLmsux)`)
        res = corpus.chunk("page").select(
            "ordinal", rcount=RegexCount(SimpleStringify(), r"(?i)o")
        )

        # One page has 3 `o`s (including capitals), the other has 6
        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "rcount": 4},
            {"ordinal": 2, "rcount": 6},
        ]

        # Test `flag` parameters
        res = corpus.chunk("page").select(
            "ordinal", rcount=RegexCount(SimpleStringify(), r"o", flags=re.IGNORECASE)
        )
        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "rcount": 4},
            {"ordinal": 2, "rcount": 6},
        ]


class TestChunkOverlap:
    def test_expr(self, ocr_corpus):
        corpus = ocr_corpus

        # Set up chunks to compare against (built-in ones all already overlap...)
        res = corpus.chunk("line").select(
            "ordinal",
            overlap=ChunkOverlap(
                chunk_b=RegexMatchChunk(
                    constrain_to="document", pattern="Over the <lazy>"
                ),
                agg="count",
            ),
        )

        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "overlap": 0},
            {"ordinal": 2, "overlap": 0},
            {"ordinal": 3, "overlap": 2},
            {"ordinal": 4, "overlap": 1},
            {"ordinal": 5, "overlap": 0},
            {"ordinal": 6, "overlap": 0},
            {"ordinal": 7, "overlap": 0},
            {"ordinal": 8, "overlap": 0},
        ]

        # `bool` agg tests
        res = corpus.chunk("line").select(
            "ordinal",
            overlap=ChunkOverlap(
                chunk_b=RegexMatchChunk(
                    constrain_to="document", pattern="Over the <lazy>"
                ),
                agg="bool",
            ),
        )

        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "overlap": False},
            {"ordinal": 2, "overlap": False},
            {"ordinal": 3, "overlap": True},
            {"ordinal": 4, "overlap": True},
            {"ordinal": 5, "overlap": False},
            {"ordinal": 6, "overlap": False},
            {"ordinal": 7, "overlap": False},
            {"ordinal": 8, "overlap": False},
        ]

        # `frac` agg tests
        res = corpus.chunk("line").select(
            "ordinal",
            overlap=ChunkOverlap(
                chunk_b=RegexMatchChunk(
                    constrain_to="document", pattern="Over the <lazy>"
                ),
                agg="frac",
            ),
        )

        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "overlap": 0},
            {"ordinal": 2, "overlap": 0},
            {"ordinal": 3, "overlap": 1},
            {"ordinal": 4, "overlap": 0.5},
            {"ordinal": 5, "overlap": 0},
            {"ordinal": 6, "overlap": 0},
            {"ordinal": 7, "overlap": 0},
            {"ordinal": 8, "overlap": 0},
        ]


class TestSimpleStringify:
    def test_expr(self, ocr_corpus):
        corpus = ocr_corpus

        res = corpus.chunk("page").select("ordinal", t=SimpleStringify(delimiter="?"))

        assert len(res) == 2

        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "t": "The?(quick)?[brown]?fox?jumps!?Over?the?<lazy>?dog"},
            {"ordinal": 2, "t": "The?~groovy?minute!?dog?bounds?UPON?the?sleepy?fox"},
        ]

        # Verify ordering
        reordered = corpus.chunk("page")
        reordered.chunks = reordered.chunks[::-1]
        res = reordered.select("ordinal", t=SimpleStringify(delimiter="?"))

        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "t": "The?(quick)?[brown]?fox?jumps!?Over?the?<lazy>?dog"},
            {"ordinal": 2, "t": "The?~groovy?minute!?dog?bounds?UPON?the?sleepy?fox"},
        ]


class TestChunkDelimitedStringify:
    def test_expr(self, ocr_corpus):
        corpus = ocr_corpus

        res = corpus.chunk("page").select(
            "ordinal",
            t=ChunkDelimitedStringify(
                chunk_delimiters=[("paragraph", "¶"), ("line", "•")],
                atom_delimiter="?",
            ),
        )

        assert len(res) == 2

        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "t": "The?(quick)•[brown]?fox?jumps!¶Over?the•<lazy>?dog"},
            {"ordinal": 2, "t": "The?~groovy•minute!?dog?bounds¶UPON?the•sleepy?fox"},
        ]

        # Verify ordering
        reordered = corpus.chunk("page")
        reordered.chunks = reordered.chunks[::-1]
        res = reordered.select(
            "ordinal",
            t=ChunkDelimitedStringify(
                chunk_delimiters=[("paragraph", "¶"), ("line", "•")],
                atom_delimiter="?",
            ),
        )

        assert res.sort_by("ordinal").to_pylist() == [
            {"ordinal": 1, "t": "The?(quick)•[brown]?fox?jumps!¶Over?the•<lazy>?dog"},
            {"ordinal": 2, "t": "The?~groovy•minute!?dog?bounds¶UPON?the•sleepy?fox"},
        ]
