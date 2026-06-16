import pandas as pd

books = pd.read_csv(
    "data/processed/books_clean.csv"
)

print(
    f"Total de livros: {len(books):,}"
)

print("\nDistribuição:")

for threshold in [10, 50, 100, 500, 1000, 5000]:

    count = len(
        books[
            books["numRatings"] >= threshold
        ]
    )

    percentage = (
        count / len(books)
    ) * 100

    print(
        f"numRatings >= {threshold:<5} "
        f"-> {count:>6,} livros "
        f"({percentage:.2f}%)"
    )

print("\nResumo estatístico:")

print(
    books["numRatings"]
    .describe()
)