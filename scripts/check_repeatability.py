"""Проверяет, что два запуска pipeline на одних данных дают идентичные предсказания."""
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.features import make_pipeline, load_data, clean_data


def hash_array(arr: np.ndarray) -> str:
    """MD5 от непрерывного float64-массива."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.md5(arr.tobytes()).hexdigest()


def main():
    df = clean_data(load_data())
    X = df.drop(columns=["high_rating"])
    y = df["high_rating"].values

    model = make_pipeline(
        LogisticRegression(C=1, max_iter=2000, solver="lbfgs", random_state=42)
    )

    # Первый запуск
    model.fit(X, y)
    pred1 = model.predict_proba(X)[:, 1]
    h1 = hash_array(pred1)

    # Второй запуск (тот же объект pipeline, те же данные, тот же seed)
    model.fit(X, y)
    pred2 = model.predict_proba(X)[:, 1]
    h2 = hash_array(pred2)

    print(f"Hash run 1: {h1}")
    print(f"Hash run 2: {h2}")
    print(f"Repeatable:  {h1 == h2}")
    print(f"Max |diff|:  {np.max(np.abs(pred1 - pred2)):.3e}")


if __name__ == "__main__":
    main()