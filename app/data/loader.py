from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/raw")


def load_books() -> pd.DataFrame:
    books_path = DATA_DIR / "books.csv"

    if not books_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {books_path}"
        )

    return pd.read_csv(books_path)


def load_ratings() -> pd.DataFrame:
    ratings_path = DATA_DIR / "ratings.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ratings_path}"
        )

    return pd.read_csv(ratings_path, nrows=10)