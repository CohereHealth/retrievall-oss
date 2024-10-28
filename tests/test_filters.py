from retrievall.filters import (
    TopK,
    Threshold,
    EqualTo,
)


class TestTopK:
    def test_filter(self, ocr_corpus):
        corpus = ocr_corpus

        res = corpus.chunk("line").filter(TopK("width", 2)).select("ordinal")

        # Widest lines are lines 2 and 6
        assert res.sort_by("ordinal")["ordinal"].to_pylist() == [2, 6]


class TestThreshold:
    def test_filter(self, ocr_corpus):
        corpus = ocr_corpus

        res = (
            corpus.chunk("line").filter(Threshold("width", ">", 110)).select("ordinal")
        )

        # Lines 2 and 6 have a `width` > 110
        assert res.sort_by("ordinal")["ordinal"].to_pylist() == [2, 6]

        res = (
            corpus.chunk("line").filter(Threshold("width", ">=", 110)).select("ordinal")
        )

        # Lines 1, 2, 5, and 6 have a `width` >= 110
        assert res.sort_by("ordinal")["ordinal"].to_pylist() == [1, 2, 5, 6]


class TestEqualTo:
    def test_chunker(self, ocr_corpus):
        corpus = ocr_corpus
        res = corpus.chunk("line").filter(EqualTo("width", [80, 100])).select("ordinal")

        # Lines 3, 4, 7, and 8 have a width of 80 or 100
        assert res.sort_by("ordinal")["ordinal"].to_pylist() == [3, 4, 7, 8]
