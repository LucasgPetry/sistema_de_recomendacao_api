from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def health_check():
    return {
        "project": "Book Recommendation System",
        "status": "running",
        "version": "1.0.0"
    }