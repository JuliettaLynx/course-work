import pandas as pd
from pathlib import Path
import yaml

def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    
    train_ratio = params["split"]["train_ratio"] # 0.65
    date_col = params["split"]["date_col"]

    RAW_PATH = Path("data/raw/interactions_large.csv")
    OUT_TRAIN = Path("data/processed/train_split.csv")
    INCOMING_DIR = Path("data/incoming")
    
    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка данных...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    
    # 1. Очистка и фильтрация (оставляем только прочитанные с оценкой)
    df = df.drop_duplicates(subset=["user_id", "book_id"])
    df = df[(df["is_read"] == 1) & df["rating"].between(1, 5)].copy()
    df["high_rating"] = (df["rating"] > 4).astype(int)
    
    # 2. Парсинг дат
    df[date_col] = pd.to_datetime(df[date_col], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    total_valid_rows = len(df)
    print(f"Всего валидных записей для разбиения: {total_valid_rows}")

    # 3. Создание признака года
    df['year'] = df[date_col].dt.year

    # 4. Разбиение (накопление по годам до train_ratio)
    train_years = []
    current_rows = 0
    threshold = train_ratio * total_valid_rows
    
    for year in sorted(df['year'].unique()):
        train_years.append(year)
        current_rows += len(df[df['year'] == year])
        if current_rows >= threshold:
            break

    train_df = df[df['year'].isin(train_years)].copy()
    test_df = df[~df['year'].isin(train_years)].copy()

    # 5. Сохранение Train
    train_df = train_df.drop(columns=['year'])
    train_df.to_csv(OUT_TRAIN, index=False)

    print(f"Обучающая выборка (Train) сохранена в: {OUT_TRAIN.name}")
    print(f"  -> Годы в Train: {train_years[0]} - {train_years[-1]}")
    print(f"  -> Строк в Train: {len(train_df)} ({len(train_df)/total_valid_rows*100:.2f}%)")

    print("Созданные порции новых данных (Incoming):")
    # 6. Сохранение порций
    batches = sorted(test_df['year'].unique())
    for i, year in enumerate(batches, start=1):
        batch_df = test_df[test_df['year'] == year].drop(columns=['year'])
        batch_path = INCOMING_DIR / f"batch_{i:02d}_{year}.csv"
        batch_df.to_csv(batch_path, index=False)
        
        pct = (len(batch_df) / total_valid_rows) * 100
        print(f"  -> {batch_path.name}: {len(batch_df)} строк ({pct:.2f}%)")

if __name__ == "__main__":
    main()