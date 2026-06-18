import pandas as pd

from app.recommender.hybrid import (
    hybrid_recommendation,
)

from app.recommender.content_based import (
    train_content_model,
)

from app.recommender.collaborative import (
    train_collaborative_model,
)

print("Carregando datasets...")

books = pd.read_csv(
    "data/processed/books_clean.csv"
)

ratings = pd.read_csv(
    "data/processed/ratings_harmonized.csv"
)

print("Treinando TF-IDF...")

tfidf_vectorizer, tfidf_matrix = (
    train_content_model(books)
)

print(
    f"Matriz TF-IDF: {tfidf_matrix.shape}"
)

print("Treinando SVD...")

svd_model = train_collaborative_model(
    ratings
)

print("SVD treinado.")

print(
    f"Livros: {len(books):,}"
)

print(
    f"Ratings: {len(ratings):,}"
)

result = hybrid_recommendation(
    user_id=9880,
    books_df=books,
    ratings_df=ratings,
    tfidf_matrix=tfidf_matrix,
    collaborative_model=svd_model,
)

print("\nResultado:")
print("\nLivro favorito:")
print(
    result["favorite_title"]
)

print("\nCONTENT:")
for book in result[
    "content_candidates"
]:
    print(f"- {book}")

print("\nCOLLABORATIVE:")
for book in result[
    "collaborative_candidates"
]:
    print(f"- {book}")

print("\nHYBRID:")

for rec in result[
    "hybrid_recommendations"
]:

    print(
        f"\nTítulo: {rec['title']}"
    )

    print(
        f"Fonte: {rec['source']}"
    )

    print(
        f"Motivo: {rec['reason']}"
    )

    print(
        f"Score: {rec['score']}"
    )