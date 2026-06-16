import pandas as pd

ratings = pd.read_csv(
    "data/processed/ratings_harmonized.csv"
)

user_counts = (
    ratings["user_id"]
    .value_counts()
)

print(
    user_counts.head(20)
)

print()

print(
    user_counts.describe()
)