from fastapi import APIRouter

from app.data import data_store
from app.recommender import model_store

from app.recommender.collaborative import (
    recommend_for_user,
)

router = APIRouter(
    prefix="/recommend",
    tags=["Collaborative"],
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
        recommendations[
            [
                "title",
                "author",
                "predicted_rating",
            ]
        ]
        .to_dict(
            orient="records"
        )
    )