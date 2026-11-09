import pandas as pd
from pathlib import Path

def main():
    raw_path = Path("data/raw/interactions_large.csv")
    if not raw_path.exists():
        print(f"Ошибка: файл {raw_path} не найден.")
        return

    print("Загрузка данных для разведки...")
    # Загружаем нужные колонки для фильтрации
    df = pd.read_csv(raw_path, usecols=["date_added", "is_read", "rating"])
    
    print("Фильтрация данных (оставляем только прочитанные с оценкой 1-5)...")
    df = df[(df["is_read"] == 1) & df["rating"].between(1, 5)].copy()
    
    df["date_added"] = pd.to_datetime(
        df["date_added"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True
    )
    df = df.dropna(subset=["date_added"])

    df['year'] = df['date_added'].dt.year
    df['half_year'] = df['year'].astype(str) + '-H' + ((df['date_added'].dt.month - 1) // 6 + 1).astype(str)

    print(f"\nВсего валидных записей после фильтрации: {len(df)}")

    # 1. Анализ по годам
    year_counts = df['year'].value_counts().sort_index()
    year_cum_pct = (year_counts.cumsum() / len(df)) * 100
    
    print("\n--- Накопительный процент по ГОДАМ ---")
    print(pd.DataFrame({'Кол-во': year_counts, 'Накопительный_%': year_cum_pct.round(2)}))

if __name__ == "__main__":
    main()