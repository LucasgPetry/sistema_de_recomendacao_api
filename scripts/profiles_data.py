from pathlib import Path

import pandas as pd

BOOKS_PATH = Path("data/raw/books.csv")
RATINGS_PATH = Path("data/processed/ratings_sample.csv")

books = pd.read_csv(BOOKS_PATH)
ratings = pd.read_csv(RATINGS_PATH)

print("\n=== BOOKS ===")

print(f"Linhas: {len(books):,}")
print(f"Colunas: {len(books.columns)}")

print("\nValores nulos:")
print(books.isnull().sum())

print("\nDuplicados:")
print(books.duplicated().sum())

print("\n=== RATINGS ===")

print(f"Linhas: {len(ratings):,}")

print(
    f"Usuários únicos: "
    f"{ratings['user_id'].nunique():,}"
)

print(
    f"Livros únicos: "
    f"{ratings['book_id'].nunique():,}"
)

print("\nDistribuição das notas:")
print(
    ratings["rating"]
    .value_counts()
    .sort_index()
)