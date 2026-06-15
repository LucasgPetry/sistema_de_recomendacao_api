from pathlib import Path

import pandas as pd

books = pd.read_csv(
    Path("data/raw/books.csv")
)

ratings = pd.read_csv(
    Path("data/processed/ratings_harmonized.csv")
)

valid = ratings["book_id"].isin(
    books["bookId"]
)

print(
    f"Taxa de correspondência: "
    f"{valid.mean():.2%}"
)