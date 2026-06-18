from app.data.loader import (
    load_books,
    load_ratings,
)

from app.recommender.content_based import (
    train_content_model,
)

from app.recommender.collaborative import (
    train_collaborative_model,
)

from app.recommender.hybrid import (
    hybrid_recommendation,
)

print("Carregando dados...")

books = load_books()
ratings = load_ratings()

tfidf_matrix = (
    train_content_model(
        books
    )
)

svd_model = (
    train_collaborative_model(
        ratings
    )
)

test_users = [
    173,      # 1 avaliação
    1073,     # ~35 avaliações
    9880,     # 66 avaliações
]

for user_id in test_users:

    print("\n" + "=" * 60)

    result = hybrid_recommendation(
        user_id=user_id,
        books_df=books,
        ratings_df=ratings,
        tfidf_matrix=tfidf_matrix,
        collaborative_model=svd_model,
    )

    print(
        f"Usuário: {user_id}"
    )

    print(
        f"Estratégia: "
        f"{result['strategy']}"
    )