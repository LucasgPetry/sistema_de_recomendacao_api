from fastapi import APIRouter
from fastapi import HTTPException

from app.recommender import model_store
from app.recommender.content_based import (
    recommend_by_title,
)

router = APIRouter(
    prefix="/recommend",
    tags=["Recommendations"],
)

@router.get("/content/{title}")
def recommend_content(
    title: str,
    top_k: int = 10,
):

    try:

        recommendations = (
            recommend_by_title(
                title=title,
                books=model_store.books_df,
                tfidf_matrix=model_store.tfidf_matrix,
                top_k=top_k,
            )
        )

        return {
            "query": title,
            "recommendations": (
                recommendations
                .to_dict(
                    orient="records"
                )
            ),
        }

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )