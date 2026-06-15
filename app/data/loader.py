from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")


def load_books() -> pd.DataFrame:
    books_path = DATA_DIR / "books_clean.csv"

    if not books_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {books_path}"
        )

    return pd.read_csv(books_path)


def load_ratings() -> pd.DataFrame:
    ratings_path = DATA_DIR / "ratings_harmonized.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {ratings_path}"
        )

    return pd.read_csv(ratings_path)