import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path

def run_chi2(name, df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    if contingency_table.empty:
        print(f"\n--- {name} ---\nНедостаточно данных для проверки.")
        return

    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1))) if n > 0 else 0
    
    verdict = "Подтверждена" if p < 0.05 else "Отклонена"
    print(f"\n--- {name} ---")
    print(f"p-value: {p:.6f}")
    print(f"V Крамера: {cramers_v:.3f}")
    print(f"Вывод (α=0.05): {verdict}")

def test_hypotheses():
    train_path = Path("data/processed/train_split.csv")
    meta_path = Path("data/raw/books_metadata_small.csv")
    
    if not train_path.exists() or not meta_path.exists():
        print("Ошибка: не найдены файлы данных.")
        return

    df = pd.read_csv(train_path)
    df_clean = df[(df["is_read"] == 1) & df["rating"].between(1, 5)].copy()
    df_clean["high_rating"] = (df_clean["rating"] > 4).astype(int)
    
    meta = pd.read_csv(meta_path)
    merged = df_clean.merge(meta, on="book_id", how="inner")

    # ГИПОТЕЗА 1: Строгость пользователя
    user_stats = df_clean.groupby("user_id")["rating"].mean().reset_index()
    user_stats.rename(columns={"rating": "user_avg_rating"}, inplace=True)
    merged_gl = merged.merge(user_stats, on="user_id", how="left")
    
    merged_gl['user_type'] = np.where(merged_gl['user_avg_rating'] > 4.2, 'loyal', 'strict')
    run_chi2("Г1: Строгость юзера (лояльные vs строгие)", merged_gl, 'user_type', 'high_rating')

    # ГИПОТЕЗА 2: Соответствие ожиданиям (разница > 0.5 и < -0.5)
    merged_gl["rating_diff"] = merged_gl["average_rating"] - merged_gl["user_avg_rating"]
    merged_gl['diff_cat'] = np.where(merged_gl['rating_diff'] > 0.5, 'positive', 'negative')
    
    # Оставляем только крайние случаи для чистоты эксперимента
    mask = (merged_gl['rating_diff'] > 0.5) | (merged_gl['rating_diff'] < -0.5)
    run_chi2("Г2: Соответствие ожиданиям (разница рейтингов)", merged_gl[mask], 'diff_cat', 'high_rating')

    # ГИПОТЕЗА 3: Популярность книги
    merged['is_popular'] = np.where(merged['ratings_count'] > 100, 'popular', 'unpopular')
    run_chi2("Г3: Популярность книги (оценок > 100)", merged, 'is_popular', 'high_rating')

if __name__ == "__main__":
    test_hypotheses()