from fastapi import APIRouter

from app.data import data_store

router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"]
)


@router.get("/status")
def dataset_status():

    return {
        "books_loaded": data_store.books_df is not None,
        "ratings_loaded": data_store.ratings_df is not None,
    }