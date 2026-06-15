from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.data.loader import load_books, load_ratings
from app.data import data_store

from app.recommender.content_based import (
    initialize_content_model,
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:

        print("Iniciando carga dos livros...")
        data_store.books_df = load_books()
        print("Livros carregados")

        print("Iniciando carga dos ratings...")
        data_store.ratings_df = load_ratings()
        print("Ratings carregados")

        print(
            f"Books carregados: {len(data_store.books_df)}"
        )

        print(
            f"Ratings carregados: {len(data_store.ratings_df)}"
        )

        print("Treinando modelo TF-IDF...")

        initialize_content_model(
            data_store.books_df
        )

        print("Modelo TF-IDF carregado.")

    except Exception as e:

        print(f"ERRO NO STARTUP: {e}")
        raise

    yield


app = FastAPI(
    title="Book Recommendation API",
    description="Sistema híbrido de recomendação de livros",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router) 

