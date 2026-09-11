"""创建垃圾邮件分类流水线。"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler


NUMERIC_FEATURES = [
    "text_length",
    "subject_length",
    "url_count",
    "recipient_count",
    "attachment_count",
    "suspicious_attachment_count",
    "exclamation_count",
    "uppercase_ratio",
    "has_html",
    "reply_to_mismatch",
]


def build_model(
    max_word_features: int = 80_000,
    max_char_features: int = 120_000,
    regularization_c: float = 1.0,
):
    features = ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_word_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
                "text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=max_char_features,
                    sublinear_tf=True,
                ),
                "text",
            ),
            ("metadata", MaxAbsScaler(), NUMERIC_FEATURES),
        ],
        sparse_threshold=1.0,
    )
    classifier = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2_000,
        random_state=42,
    )
    return Pipeline([("features", features), ("classifier", classifier)])
