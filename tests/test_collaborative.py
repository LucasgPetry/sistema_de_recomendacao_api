from app.data.loader import (
    load_books,
    load_ratings,
)

from app.recommender.collaborative import (
    train_collaborative_model,
    recommend_for_user,
)


def test_collaborative_recommendation():

    books = load_books()

    ratings = load_ratings()

    model = (
        train_collaborative_model(
            ratings
        )
    )

    recommendations = (
        recommend_for_user(
            user_id=9880,
            books_df=books,
            ratings_df=ratings,
            model=model,
            top_k=5,
        )
    )

    assert len(recommendations) > 0