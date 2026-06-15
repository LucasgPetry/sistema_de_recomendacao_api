from pathlib import Path

import pandas as pd

BOOKS_PATH = Path("data/raw/books.csv")
RATINGS_PATH = Path("data/processed/ratings_sample.csv")

print("Carregando datasets...")

books = pd.read_csv(BOOKS_PATH)
ratings = pd.read_csv(RATINGS_PATH)

print("Datasets carregados.")

books_ids = set(books["bookId"])

ratings["book_exists"] = ratings["book_id"].isin(
    books_ids
)

print("\n=== MAPEAMENTO ===")

mapping_stats = (
    ratings["book_exists"]
    .value_counts()
)

print(mapping_stats)

matched = ratings["book_exists"].sum()

total = len(ratings)

print(
    f"\nTaxa de correspondência: "
    f"{matched / total:.2%}"
)

print(
    f"Livros catálogo: "
    f"{books['bookId'].nunique():,}"
)

print(
    f"Livros avaliados: "
    f"{ratings['book_id'].nunique():,}"
)

print("\nExemplos não encontrados:")

print(
    ratings.loc[
        ~ratings["book_exists"],
        "book_id"
    ].head(20)
)