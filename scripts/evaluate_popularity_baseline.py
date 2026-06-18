import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)

from math import sqrt


print("Carregando ratings...")

ratings = pd.read_csv(
    "data/processed/ratings_harmonized.csv"
)

print(
    f"Ratings: {len(ratings):,}"
)

global_mean = ratings[
    "rating"
].mean()

print(
    f"\nMédia global: "
    f"{global_mean:.4f}"
)

y_true = ratings["rating"]

y_pred = [
    global_mean
] * len(ratings)

rmse = sqrt(
    mean_squared_error(
        y_true,
        y_pred,
    )
)

mae = mean_absolute_error(
    y_true,
    y_pred,
)

print("\n=== RESULTADOS ===")

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"MAE: {mae:.4f}"
)