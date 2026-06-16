import pandas as pd

from app.recommender.collaborative import (
    train_collaborative_model,
    recommend_for_user,
)


def main():

    print("Carregando datasets...")

    books = pd.read_csv(
        "data/processed/books_clean.csv"
    )

    ratings = pd.read_csv(
        "data/processed/ratings_harmonized.csv"
    )

    print(
        f"Livros: {len(books):,}"
    )

    print(
        f"Ratings: {len(ratings):,}"
    )

    model = train_collaborative_model(
        ratings
    )

    test_user = 9880

    print(
        f"\nTestando usuário: "
        f"{test_user}"
    )

    recommendations = (
        recommend_for_user(
            user_id=test_user,
            books_df=books,
            ratings_df=ratings,
            model=model,
            top_k=10,
        )
    )

    print("\nRecomendações:")

    print(
        recommendations[
            [
                "title",
                "author",
                "predicted_rating",
            ]
        ]
    )

    print("\nLivros avaliados pelo usuário:")

    user_history = ratings[
        ratings["user_id"] == test_user
    ]

    history_books = books[
        books["bookId"].isin(
            user_history["book_id"]
        )
    ]

    print(
        history_books[
            [
                "title",
                "author",
                "genres",
            ]
        ].head(20)
    )



if __name__ == "__main__":
    main()