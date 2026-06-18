import pandas as pd

from app.recommender.content_based import (
    recommend_by_title,
)

import pandas as pd

from app.recommender.collaborative import (
    recommend_for_user,
)

def get_user_favorite_book(
    user_id: int,
    ratings_df: pd.DataFrame,
):
    """
    Retorna o book_id favorito do usuário
    (maior nota atribuída).
    """

    user_ratings = ratings_df[
        ratings_df["user_id"] == user_id
    ]

    if user_ratings.empty:
        return None

    best_rating = (
        user_ratings["rating"]
        .max()
    )

    favorites = user_ratings[
        user_ratings["rating"] == best_rating
    ]

    return favorites.iloc[0]["book_id"]


def get_book_title(
    book_id: str,
    books_df: pd.DataFrame,
):
    """
    Converte book_id em título.
    """

    book = books_df[
        books_df["bookId"] == book_id
    ]

    if book.empty:
        return None

    return book.iloc[0]["title"]

def get_content_candidates(
    favorite_title,
    books_df,
):
    recommendations = recommend_by_title(
    title=favorite_title,
    books_df=books_df,
    top_k=10,
)
    return recommendations


def hybrid_recommendation(
    user_id: int,
    books_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    tfidf_matrix,
    collaborative_model=None,
):
    """
    Primeira versão do híbrido.

    Apenas valida:
    - usuário
    - livro favorito
    - título favorito
    """

    favorite_book_id = get_user_favorite_book(
        user_id=user_id,
        ratings_df=ratings_df,
    )

    if favorite_book_id is None:
        return {
            "user_id": user_id,
            "favorite_book_id": None,
            "favorite_title": None,
            "message": "Usuário sem avaliações."
        }

    favorite_title = get_book_title(
        book_id=favorite_book_id,
        books_df=books_df,
    )

    content_candidates = (
    get_content_candidates(
        favorite_title,
        books_df,
        tfidf_matrix,
    )
)
    
    collaborative_candidates = (
    get_collaborative_candidates(
        user_id,
        books_df,
        ratings_df,
        collaborative_model,
    )
)
    
    final_recommendations = (
    merge_recommendations(
        content_candidates,
        collaborative_candidates,
        top_k=10,
    )
)

    return {
    "user_id": user_id,
    "favorite_title": favorite_title,

    "content_candidates": (
        content_candidates["title"]
        .tolist()
    ),

    "collaborative_candidates": (
        collaborative_candidates["title"]
        .tolist()
    ),

    "hybrid_recommendations": (
        final_recommendations
    ),
}

def get_content_candidates(
    favorite_title: str,
    books_df: pd.DataFrame,
    tfidf_matrix,
):
    recommendations = recommend_by_title(
        title=favorite_title,
        books=books_df,
        tfidf_matrix=tfidf_matrix,
        top_k=10,
    )

    return recommendations

def get_collaborative_candidates(
    user_id: int,
    books_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    collaborative_model,
):
    recommendations = recommend_for_user(
        user_id=user_id,
        books_df=books_df,
        ratings_df=ratings_df,
        model=collaborative_model,
        top_k=10,
    )

    return recommendations

def merge_recommendations(
    content_candidates,
    collaborative_candidates,
    top_k=10,
):
    """
    Combina recomendações Content-Based e Collaborative
    preservando origem, motivo e score.
    """

    merged = []

    content_records = []

    for _, row in content_candidates.iterrows():

        content_records.append(
            {
                "title": row["title"],
                "source": "content",
                "reason": "Similar ao livro favorito",
                "score": round(
                    float(
                        row["similarity_score"]
                    ),
                    4,
                ),
            }
        )

    collaborative_records = []

    for _, row in collaborative_candidates.iterrows():

        collaborative_records.append(
            {
                "title": row["title"],
                "source": "collaborative",
                "reason": (
                    "Usuários semelhantes gostaram"
                ),
                "score": round(
                    float(
                        row["predicted_rating"]
                    ),
                    4,
                ),
            }
        )

    max_size = max(
        len(content_records),
        len(collaborative_records),
    )

    for i in range(max_size):

        if i < len(content_records):
            merged.append(
                content_records[i]
            )

        if i < len(collaborative_records):
            merged.append(
                collaborative_records[i]
            )

    seen = set()
    unique = []

    for recommendation in merged:

        if (
            recommendation["title"]
            not in seen
        ):

            unique.append(
                recommendation
            )

            seen.add(
                recommendation["title"]
            )

    return unique[:top_k]