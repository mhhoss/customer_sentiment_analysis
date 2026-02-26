from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15
NROWS = None


def resolve_data_path(project_root: str | Path | None = None) -> Path:
    if project_root is None:
        base = Path(__file__).resolve().parent.parent
    else:
        base = Path(project_root).resolve()

    return (base / "data" / "raw" / "amazon_reviews_us_Digital_Software_v1_00.tsv").resolve()


def prepare_label(df: pd.DataFrame, output_col: str = "label") -> pd.DataFrame:
    """
    Create/normalize target label column.
    Priority:
      1) existing `label`
      2) existing `sentiment`
      3) derived from `star_rating` -> negative/neutral/positive
    """
    out = df.copy()

    if "label" in out.columns:
        out[output_col] = out["label"].astype(str)
        return out

    if "star_rating" not in out.columns:
        raise ValueError("No target source found. Expected one of: label, sentiment, star_rating")

    stars = pd.to_numeric(out["star_rating"], errors="coerce")
    out[output_col] = pd.cut(
        stars,
        bins=[0, 2, 3, 5],
        labels=["negative", "neutral", "positive"],
        include_lowest=True,
    ).astype(str)
    out = out[out[output_col] != "nan"].copy()
    return out


def pick_text_column(df: pd.DataFrame) -> str:
    if "normalized_review" in df.columns:
        return "normalized_review"
    raise ValueError("text column not found. (normalized_review)")


def split_data(
    df: pd.DataFrame,
    text_col: str,
    label_col: str = "label",
    valid_size: float = VALID_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:

    x = df[text_col].fillna("").astype(str)
    y = df[label_col].astype(str)

    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    valid_ratio_on_train_valid = valid_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid,
        y_train_valid,
        test_size=valid_ratio_on_train_valid,
        random_state=random_state,
        stratify=y_train_valid,
    )  # keep class distribution balanced in train/valid

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def build_tfidf_baseline(
    x_train: pd.Series,
    x_valid: pd.Series,
    x_test: pd.Series,
    min_df: int = 5,
    max_features: int = 80_000,
) -> Tuple[TfidfVectorizer, object, object, object]:

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        sublinear_tf=True,
        max_features=max_features,
        lowercase=False,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_valid_tfidf = vectorizer.transform(x_valid)
    x_test_tfidf = vectorizer.transform(x_test)

    return vectorizer, x_train_tfidf, x_valid_tfidf, x_test_tfidf



if __name__ == "__main__":
    path = resolve_data_path()
    df_raw = pd.read_csv(path, sep="\t", encoding="utf-8", nrows=NROWS)
    df = prepare_label(df_raw)
    text_col_name = pick_text_column(df)
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(df, text_col=text_col_name)
    vec, x_tr, x_va, x_te = build_tfidf_baseline(x_train, x_valid, x_test)

    print(f"Data: {path}")
    print(f"Text col: {text_col_name}")
    print(f"Split -> train={len(x_train)}, valid={len(x_valid)}, test={len(x_test)}")
    print(f"TF-IDF -> train={x_tr.shape}, valid={x_va.shape}, test={x_te.shape}, vocab={len(vec.vocabulary_)}")
