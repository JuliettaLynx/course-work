import pandas as pd

def analyze_quality():
    print("1. Анализ целевой переменной...")
    df = pd.read_csv("data/processed/train_split.csv")
    
    # Оставляем только прочитанные книги
    df_clean = df[(df["is_read"] == 1) & df["rating"].between(1, 5)].copy()
    df_clean["high_rating"] = (df_clean["rating"] > 4).astype(int)
    
    # Баланс классов
    print("\nРаспределение целевой переменной (high_rating):")
    print(df_clean["high_rating"].value_counts(normalize=True))
    
    print("\n2. Описательные статистики для interactions:")
    print(df_clean[["rating"]].describe())

    print("\n3. Анализ качества метаданных книг...")
    meta = pd.read_csv("data/raw/books_metadata_small.csv")
    
    print("\nПроцент пропусков в признаках книг:")
    print((meta.isna().mean() * 100).round(2))
    
    print("\nОписательные статистики метаданных:")
    print(meta[["average_rating", "num_pages", "publication_year", "ratings_count"]].describe())

if __name__ == "__main__":
    analyze_quality()