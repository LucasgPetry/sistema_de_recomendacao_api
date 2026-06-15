import pandas as pd

df = pd.read_csv(
    "data/processed/books_clean.csv"
)

print(df[[
    "title",
    "author",
    "genres",
    "description"
]].head(5))