from pathlib import Path

import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD

RATINGS_PATH = Path(
    "data/processed/ratings_harmonized.csv"
)

ratings = pd.read_csv(
    RATINGS_PATH
)

ratings = ratings[
    [
        "user_id",
        "book_id",
        "rating"
    ]
]

reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(
    ratings,
    reader
)

trainset = data.build_full_trainset()

model = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=42,
)

print("Treinando SVD...")

model.fit(trainset)

print("Treinamento concluído.")

sample_user = ratings[
    "user_id"
].iloc[0]

sample_book = ratings[
    "book_id"
].iloc[0]

prediction = model.predict(
    sample_user,
    sample_book
)

print(prediction)