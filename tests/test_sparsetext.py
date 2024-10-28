import pytest
from retrievall.exprs import SimpleStringify
from retrievall.sparsetext import (
    Tfidf,
)


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
