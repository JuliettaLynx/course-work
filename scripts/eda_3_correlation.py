import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation():
    df = pd.read_csv("data/processed/train_split.csv")
    meta = pd.read_csv("data/raw/books_metadata_small.csv")
    merged = df.merge(meta, on="book_id", how="inner")
    
    cols = ["high_rating", "average_rating", "num_pages", "publication_year", "ratings_count"]
    corr_matrix = merged[cols].corr(method='spearman')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Матрица корреляции Спирмена")
    plt.tight_layout()
    plt.savefig("reports/eda_correlation.png")

if __name__ == "__main__":
    plot_correlation()