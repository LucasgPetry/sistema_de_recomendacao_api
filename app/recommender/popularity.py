import pandas as pd


def recommend_popular_books(
    books_df: pd.DataFrame,
    top_k: int = 10,
):

    recommendations = (
        books_df
        .sort_values(
            by=[
                "numRatings",
                "rating",
            ],
            ascending=False,
        )
        .head(top_k)
        .copy()
    )

    recommendations["source"] = (
        "popularity"
    )

    recommendations["reason"] = (
        "Livro popular entre os usuários"
    )

    recommendations["score"] = (
        recommendations["numRatings"]
    )

    return recommendations[
        [
            "title",
            "author",
            "source",
            "reason",
            "score",
        ]
    ]