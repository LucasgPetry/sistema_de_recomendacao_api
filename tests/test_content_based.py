from app.data.loader import load_books

from app.recommender.content_based import (
    train_content_model,
    recommend_by_title,
)


def test_content_recommendation():

    books = load_books()

    _, tfidf_matrix = (
        train_content_model(
            books
        )
    )

    recommendations = (
        recommend_by_title(
            title="The Hunger Games",
            books=books,
            tfidf_matrix=tfidf_matrix,
            top_k=5,
        )
    )

    assert len(recommendations) > 0

    assert "title" in recommendations.columns

    assert (
        recommendations.iloc[0]["title"]
        != "The Hunger Games"
    )