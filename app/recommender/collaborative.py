import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD


def train_collaborative_model(
    ratings: pd.DataFrame,
):
    """
    Treina o modelo SVD utilizando
    user_id, book_id e rating.
    """

    reader = Reader(
        rating_scale=(1, 5)
    )

    data = Dataset.load_from_df(
        ratings[
            [
                "user_id",
                "book_id",
                "rating",
            ]
        ],
        reader,
    )

    trainset = data.build_full_trainset()

    model = SVD(
        n_factors=50,
        n_epochs=20,
        random_state=42,
    )

    print("Treinando modelo colaborativo (SVD)...")

    model.fit(trainset)

    print("Modelo colaborativo treinado.")

    return model


def recommend_for_user(
    user_id,
    books_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    model,
    top_k: int = 10,
):
    """
    Recomenda livros para um usuário
    utilizando Collaborative Filtering.
    """

    # Livros já avaliados pelo usuário
    rated_books = set(
        ratings_df[
            ratings_df["user_id"] == user_id
        ]["book_id"]
    )

    # Catálogo completo
    all_books = set(
        books_df["bookId"]
    )

    # Livros candidatos
    candidate_books = (
        all_books - rated_books
    )

    print(
        f"Usuário {user_id} avaliou "
        f"{len(rated_books)} livros"
    )

    print(
        f"Livros candidatos: "
        f"{len(candidate_books)}"
    )

    predictions = []

    for book_id in candidate_books:

        estimated_rating = model.predict(
            user_id,
            book_id,
        ).est

        predictions.append(
            (
                book_id,
                estimated_rating,
            )
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    top_predictions = predictions[:top_k]

    recommended_ids = [
        book_id
        for book_id, _
        in top_predictions
    ]

    recommendations = books_df[
        books_df["bookId"].isin(
            recommended_ids
        )
    ].copy()

    score_map = {
        book_id: score
        for book_id, score
        in top_predictions
    }

    recommendations[
        "predicted_rating"
    ] = recommendations[
        "bookId"
    ].map(score_map)

    recommendations = (
        recommendations.sort_values(
            by="predicted_rating",
            ascending=False,
        )
    )

    return recommendations