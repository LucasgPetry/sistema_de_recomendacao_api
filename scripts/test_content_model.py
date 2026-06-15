from pathlib import Path

import pandas as pd

from app.recommender.content_based import (
    train_content_model,
    recommend_by_title,
)

BOOKS_PATH = Path(
    "data/processed/books_clean.csv"
)

print("Carregando livros...")

books = pd.read_csv(
    BOOKS_PATH
)

print(
    f"Livros carregados: "
    f"{len(books):,}"
)

print("Treinando TF-IDF...")

vectorizer, tfidf_matrix = (
    train_content_model(
        books
    )
)

print(
    f"Matriz TF-IDF: "
    f"{tfidf_matrix.shape}"
)

title = "Harry Potter and the Order of the Phoenix"

print(
    f"\nRecomendações para:"
    f"\n{title}\n"
)

recommendations = (
    recommend_by_title(
        title=title,
        books=books,
        tfidf_matrix=tfidf_matrix,
        top_k=10,
    )
)

for idx, row in enumerate(
    recommendations.itertuples(),
    start=1,
):
    print(
        f"{idx}. "
        f"{row.title} | "
        f"{row.author} | "
        f"Rating: {row.rating}"
    )