import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

RAW_INTERACTIONS = Path("data/processed/train_split.csv")
RAW_METADATA = Path("data/raw/books_metadata_small.csv")

NUMERIC_FEATURES = [
    "average_rating", "num_pages", "publication_year", "ratings_count",
    "user_avg_rating", "rating_diff", "book_age", "log_ratings_count",
]
CATEGORICAL_FEATURES = ["is_ebook", "language_code", "num_pages_bin"]

def load_data(interactions_path=RAW_INTERACTIONS, metadata_path=RAW_METADATA):
    """Загружает интеракции, применяет temporal split ≤ 2013 по date_added."""
    df = pd.read_csv(interactions_path, low_memory=False)

    if "date_added" not in df.columns:
        raise RuntimeError(
            "В train_split.csv нет столбца 'date_added'. "
            "Temporal split по году взаимодействия восстановить нельзя — "
            "пересоберите train_split.csv из сырых данных в ПР2."
        )

    dt = pd.to_datetime(df["date_added"], errors="coerce", utc=True, format="mixed")
    year = dt.dt.year
    keep = year <= 2013

    dropped = int((~keep).sum())
    print(
        f"[load_data] temporal split по 'date_added': "
        f"dropped {dropped} / {len(df)} = {dropped/len(df):.1%}, "
        f"осталось {int(keep.sum())}"
    )

    df = df[keep].copy()

    meta = pd.read_csv(metadata_path)
    return df.merge(meta, on="book_id", how="inner")

def clean_data(df):
    """Очистка согласно DQ-01–DQ-08 (ПР № 2, Таблица 5)."""
    df = df.drop_duplicates(subset=["user_id", "book_id"])
    df = df[df["is_read"] == 1]
    df = df[df["rating"].between(1, 5)].copy()
    df["high_rating"] = (df["rating"] > 4).astype(int)
    df = df.drop(
        columns=[
            "is_read", "review_id", "review_text_incomplete",
            "date_updated", "read_at", "started_at",
        ],
        errors="ignore",
    )
    # DQ-05 / DQ-06 / DQ-07: аномалии → NaN, imputer заполнит медианой по train-фолду
    df.loc[df["publication_year"] > 2026, "publication_year"] = np.nan
    df.loc[df["num_pages"] > 5000, "num_pages"] = np.nan
    df.loc[df["average_rating"] == 0, "average_rating"] = np.nan
    return df


class UserAvgRatingTransformer(BaseEstimator, TransformerMixin):
    """Средний рейтинг пользователя, оценённый только на train-фолде."""

    def __init__(self, user_col="user_id", rating_col="rating"):
        self.user_col = user_col
        self.rating_col = rating_col
        self.user_means_ = None
        self.global_mean_ = None

    def fit(self, X, y=None):
        self.global_mean_ = X[self.rating_col].mean()
        self.user_means_ = X.groupby(self.user_col)[self.rating_col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        X["user_avg_rating"] = X[self.user_col].map(self.user_means_)
        X["user_avg_rating"] = X["user_avg_rating"].fillna(self.global_mean_)
        return X.drop(columns=[self.user_col, self.rating_col], errors="ignore")


def add_features(df):
    """Новые признаки по плану ПР № 2 (Таблица 10)."""
    df = df.copy()
    df["rating_diff"] = df["average_rating"] - df["user_avg_rating"]
    df["book_age"] = 2026 - df["publication_year"]
    df["log_ratings_count"] = np.log1p(df["ratings_count"])
    bins = [0, 200, 400, 800, np.inf]
    labels = ["0-200", "200-400", "400-800", ">800"]
    df["num_pages_bin"] = pd.cut(df["num_pages"], bins=bins, labels=labels).astype(str)
    return df


def make_pipeline(model):
    """Собирает полный pipeline: user aggregate → features → prep → model."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline(steps=[
        ("user_avg", UserAvgRatingTransformer()),
        ("features", FunctionTransformer(add_features)),
        ("prep", preprocessor),
        ("model", model),
    ])