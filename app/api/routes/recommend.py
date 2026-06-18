from fastapi import APIRouter
from fastapi import HTTPException

from app.data import data_store
from app.recommender import model_store

from app.recommender.content_based import (
    recommend_by_title,
)

from app.recommender.collaborative import (
    recommend_for_user,
)

from app.recommender.hybrid import (
    hybrid_recommendation,
)

router = APIRouter(
    prefix="/recommend",
    tags=["Recommendations"]
)

@router.get("/content/{title}")
def content_recommendation(
    title: str,
):
    try:

        recommendations = (
            recommend_by_title(
                title=title,
                books=data_store.books_df,
                tfidf_matrix=(
                    model_store.tfidf_matrix
                ),
                top_k=10,
            )
        )

        return (
            recommendations
            .to_dict(
                orient="records"
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )
    
@router.get("/collaborative/{user_id}")
def collaborative_recommendation(
    user_id: int,
):
    recommendations = (
        recommend_for_user(
            user_id=user_id,
            books_df=data_store.books_df,
            ratings_df=data_store.ratings_df,
            model=model_store.svd_model,
            top_k=10,
        )
    )

    return (
        recommendations
        .to_dict(
            orient="records"
        )
    )

@router.get("/hybrid/{user_id}")
def hybrid_endpoint(
    user_id: int,
):
    return hybrid_recommendation(
        user_id=user_id,
        books_df=data_store.books_df,
        ratings_df=data_store.ratings_df,
        tfidf_matrix=model_store.tfidf_matrix,
        collaborative_model=model_store.svd_model,
    )