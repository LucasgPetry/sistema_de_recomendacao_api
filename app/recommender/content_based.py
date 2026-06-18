import pandas as pd
import re 

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

from sklearn.metrics.pairwise import (
    cosine_similarity,
)

from app.recommender import model_store

def build_content_features(
    books: pd.DataFrame,
) -> pd.Series:

    books = books.copy()

    books["genres"] = (
        books["genres"]
        .fillna("")
        .apply(clean_genres)
    )

    content = (
        books["author"].fillna("")
        + " "
        + books["genres"].fillna("")
        + " "
        + books["description"].fillna("")
    )

    return content

def train_content_model(
    books: pd.DataFrame,
):

    content = build_content_features(
        books
    )

    vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    )

    tfidf_matrix = vectorizer.fit_transform(
        content
    )

    return (
        vectorizer,
        tfidf_matrix,
    )

def find_book_index(
    books: pd.DataFrame,
    title: str,
) -> int:

    matches = books[
        books["title"]
        .str.lower()
        == title.lower()
    ]

    if matches.empty:

        raise ValueError(
            f"Livro '{title}' não encontrado."
        )

    return matches.index[0]

def recommend_by_title(
    title: str,
    books: pd.DataFrame,
    tfidf_matrix,
    top_k: int = 10,
):

    idx = find_book_index(
        books,
        title,
    )

    similarities = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix,
    ).flatten()

    similar_indices = (
        similarities.argsort()[::-1]
    )

    similar_indices = (
        similar_indices[
            1 : top_k + 20
        ]
    )

    similar_scores = similarities[
        similar_indices
    ]

    recommendations = books.iloc[
        similar_indices
    ][
        [
            "bookId",
            "title",
            "author",
            "rating",
            "likedPercent",
            "numRatings",
            "genres",
        ]
    ].copy()

    recommendations[
        "similarity_score"
    ] = similar_scores

    recommendations[
        "source"
    ] = "content"

    recommendations[
        "reason"
    ] = (
        "Similar ao livro consultado"
    )

    EXCLUDED_PATTERNS = (
        "guide|summary|companion|analysis|study|"
        "sampler|boxset|parody|workbook|notes|"
        "collection|complete novels|complete works"
    )

    recommendations = recommendations[
        ~recommendations["title"]
        .str.lower()
        .str.contains(
            EXCLUDED_PATTERNS,
            na=False,
        )
    ]

    recommendations = recommendations.sort_values(
        by="similarity_score",
        ascending=False,
    )

    return recommendations.head(top_k)

def clean_genres(text: str) -> str:

    text = re.sub(
        r"[\[\]']",
        "",
        str(text)
    )

    text = text.replace(",", " ")

    return text

def initialize_content_model(
    books: pd.DataFrame,
):

    vectorizer, tfidf_matrix = (
        train_content_model(books)
    )

    model_store.books_df = books

    model_store.tfidf_vectorizer = (
        vectorizer
    )

    model_store.tfidf_matrix = (
        tfidf_matrix
    )