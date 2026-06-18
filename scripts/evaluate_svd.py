from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate

import pandas as pd


print("Carregando ratings...")

ratings = pd.read_csv(
    "data/processed/ratings_harmonized.csv"
)

print(
    f"Ratings: {len(ratings):,}"
)

reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(
    ratings[
        [
            "user_id",
            "book_id",
            "rating",
        ]
    ],
    reader,
)

print("Executando validação cruzada...")

results = cross_validate(
    SVD(
        random_state=42
    ),
    data,
    measures=["RMSE", "MAE"],
    cv=5,
    verbose=True,
)

print("\n=== RESULTADOS ===")

print(
    f"RMSE médio: "
    f"{results['test_rmse'].mean():.4f}"
)

print(
    f"MAE médio: "
    f"{results['test_mae'].mean():.4f}"
)