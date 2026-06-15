from pathlib import Path

import pandas as pd

INPUT = Path("data/raw/books.csv")

OUTPUT = Path("data/processed/books_clean.csv")

print("Carregando catálogo...")

books = pd.read_csv(INPUT)

print(f"Livros originais: {len(books):,}")

# ==================================================
# Remover duplicados
# ==================================================

duplicates_before = books.duplicated().sum()

books = books.drop_duplicates()

print(f"Duplicados removidos: {duplicates_before:,}")

# ==================================================
# Selecionar apenas colunas relevantes
# ==================================================

books = books[
    [
        "bookId",
        "title",
        "author",
        "description",
        "genres",
        "language",
        "pages",
        "publisher",
        "publishDate",
        "rating",
        "numRatings",
        "likedPercent",
    ]
].copy()

# ==================================================
# Campos textuais
# ==================================================

books["title"] = (
    books["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

books["author"] = (
    books["author"]
    .fillna("unknown")
    .astype(str)
    .str.strip()
)

books["description"] = (
    books["description"]
    .fillna("")
    .astype(str)
    .str.strip()
)

books["genres"] = (
    books["genres"]
    .fillna("")
    .astype(str)
    .str.strip()
)

books["language"] = (
    books["language"]
    .fillna("unknown")
    .astype(str)
    .str.strip()
    .str.lower()
)

books["publisher"] = (
    books["publisher"]
    .fillna("unknown")
    .astype(str)
    .str.strip()
)

# ==================================================
# Pages
# ==================================================

books["pages"] = pd.to_numeric(
    books["pages"],
    errors="coerce"
)

median_pages = books["pages"].median()

books["pages"] = books["pages"].fillna(
    median_pages
)

# ==================================================
# Publish Date
# ==================================================

books["publishDate"] = pd.to_datetime(
    books["publishDate"],
    errors="coerce"
)

books["publishYear"] = (
    books["publishDate"]
    .dt.year
)

median_year = books["publishYear"].median()

books["publishYear"] = books[
    "publishYear"
].fillna(
    median_year
)

books["publishYear"] = books[
    "publishYear"
].astype(int)

# ==================================================
# Rating
# ==================================================

books["rating"] = pd.to_numeric(
    books["rating"],
    errors="coerce"
)

books["rating"] = books["rating"].fillna(
    books["rating"].median()
)

# ==================================================
# Num Ratings
# ==================================================

books["numRatings"] = pd.to_numeric(
    books["numRatings"],
    errors="coerce"
)

books["numRatings"] = books[
    "numRatings"
].fillna(0)

# ==================================================
# Liked Percent
# ==================================================

books["likedPercent"] = pd.to_numeric(
    books["likedPercent"],
    errors="coerce"
)

books["likedPercent"] = books[
    "likedPercent"
].fillna(
    books["likedPercent"].median()
)

# ==================================================
# Remover coluna de data original
# ==================================================

books = books.drop(
    columns=["publishDate"]
)

# ==================================================
# Salvar
# ==================================================

books.to_csv(
    OUTPUT,
    index=False
)

print(f"Livros finais: {len(books):,}")

print(f"Arquivo salvo em:\n{OUTPUT}")

print("\nResumo final:")

print(books.info())

print("\nValores nulos:")

print(
    books.isnull()
    .sum()
)