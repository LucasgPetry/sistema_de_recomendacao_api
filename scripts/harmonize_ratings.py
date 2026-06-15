from pathlib import Path

import numpy as np
import pandas as pd

BOOKS_PATH = Path("data/raw/books.csv")
RATINGS_PATH = Path("data/processed/ratings_sample.csv")

OUTPUT_PATH = Path(
    "data/processed/ratings_harmonized.csv"
)

print("Carregando datasets...")

books = pd.read_csv(BOOKS_PATH)
ratings = pd.read_csv(RATINGS_PATH)

print("Filtrando avaliações explícitas...")

ratings = ratings[
    ratings["rating"] > 0
].copy()

print(
    f"Avaliações restantes: "
    f"{len(ratings):,}"
)

print("Criando mapeamento...")

book_ids = books["bookId"].tolist()

weights = books["numRatings"].fillna(
    1
).astype(float)

weights = weights / weights.sum()

ratings["book_id"] = np.random.choice(
    book_ids,
    size=len(ratings),
    p=weights
)

ratings.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nArquivo salvo em:\n"
    f"{OUTPUT_PATH}"
)

print(
    f"Quantidade final:\n"
    f"{len(ratings):,}"
)