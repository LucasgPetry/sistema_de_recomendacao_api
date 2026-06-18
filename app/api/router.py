from fastapi import APIRouter

from app.api.routes.health import router as health_router

from app.api.routes.dataset import router as dataset_router

from app.api.routes.recommend import (
    router as recommend_router
)





api_router = APIRouter()

api_router.include_router(
    health_router,
    tags=["Health"]
) 

api_router.include_router(dataset_router)

api_router.include_router(
    recommend_router
)

