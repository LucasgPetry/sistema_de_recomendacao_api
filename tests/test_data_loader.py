from app.data.loader import load_books
from app.data.loader import load_ratings


def test_load_books():

    books = load_books()

    assert len(books) > 0
    assert "title" in books.columns
    assert "bookId" in books.columns


def test_load_ratings():

    ratings = load_ratings()

    assert len(ratings) > 0
    assert "user_id" in ratings.columns
    assert "book_id" in ratings.columns