import pyarrow as pa
from retrievall.core import Chunks, AttrExpr
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = [
    "Tfidf",
]


class Tfidf(AttrExpr):
    """
    Add a `tfidf` score column to chunks, based on their tfidf score
    relative to a `query` provided as a text string.

    Uses a sklearn `TfidfVectorizer`; keyword arguments are passed to the TfidfVectorizer.

    Parameters
    ----------
    stringifier
        An `AttrExpression` that returns one string per chunk; determine how the atoms
        will be represented as strings for TF-IDF scoring.
    query
        Text string that chunks will be scored against for similarity.
    kwargs
        `TfidfVectorizer` keyword arguments that get passed to the vectorizer.
    """

    def __init__(self, stringifier: AttrExpr, query: str, **kwargs):
        self.stringifier = stringifier
        self.query = query
        self.vectorizer = TfidfVectorizer(**kwargs)

    def __call__(self, chunks: Chunks) -> pa.Array:
        # (The stringifier is responsible for returning its strings
        # in the correct order for the input chunks.)
        strings = self.stringifier(chunks).to_pylist()

        # Apply TF-IDF
        chunk_vecs = self.vectorizer.fit_transform(strings)
        query_vec = self.vectorizer.transform([self.query])

        scores = (chunk_vecs @ query_vec.transpose()).toarray().flatten()

        return pa.array(scores)
