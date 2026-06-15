import pandas as pd

df = pd.read_csv(
    "data/processed/ratings_harmonized.csv"
)

print(df.head())

print("\nColunas:")
print(df.columns)

print("\nTipos:")
print(df.dtypes)

print("\nRatings:")
print(
    df["rating"]
    .value_counts()
    .sort_index()
)

print("\nUsuários únicos:")
print(
    df["user_id"]
    .nunique()
)

print("\nLivros únicos:")
print(
    df["book_id"]
    .nunique()
)