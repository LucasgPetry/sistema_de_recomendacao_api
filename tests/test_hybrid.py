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


def test_hybrid_recommendation():

    books = load_books()

    ratings = load_ratings()

    tfidf_matrix = (
        train_content_model(
            books
        )
    )

    model = (
        train_collaborative_model(
            ratings
        )
    )

    result = (
        hybrid_recommendation(
            user_id=9880,
            books_df=books,
            ratings_df=ratings,
            tfidf_matrix=tfidf_matrix,
            collaborative_model=model,
        )
    )

    assert result is not None