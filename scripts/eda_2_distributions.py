import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions():
    df = pd.read_csv("data/processed/train_split.csv")

    meta = pd.read_csv("data/raw/books_metadata_small.csv")
    merged = df.merge(meta, on="book_id", how="inner")
    
    # Создаем полотно для графиков
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Связь average_rating и целевой переменной
    sns.boxplot(data=merged, x="high_rating", y="average_rating", ax=axes[0])
    axes[0].set_title("Средний рейтинг книги (average_rating) vs Целевая")
    axes[0].set_xlabel("high_rating (0 - низкая, 1 - высокая)")
    axes[0].set_ylabel("average_rating")
    
    # 2. Связь ratings_count и целевой переменной
    sns.boxplot(data=merged, x="high_rating", y="ratings_count", ax=axes[1])
    axes[1].set_yscale('log') # Логарифмическая шкала из-за выбросов
    axes[1].set_title("Популярность книги (ratings_count) vs Целевая")
    axes[1].set_xlabel("high_rating (0 - низкая, 1 - высокая)")
    axes[1].set_ylabel("ratings_count (log scale)")
    
    plt.tight_layout()
    plt.savefig("reports/eda_distributions.png")

if __name__ == "__main__":
    plot_distributions()