import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def load_movielens_data(data_path: str = "data/ml-1m/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the MovieLens 1M dataset from the specified directory.

    Args:
        data_path: Path to the directory containing the MovieLens files

    Returns:
        Tuple of DataFrames (users, ratings, movies)
    """
    # Load users data
    users = pd.read_table(
        os.path.join(data_path, 'users.dat'),
        sep='::',
        header=None,
        names=['user_id', 'gender', 'age', 'occupation', 'zip'],
        engine='python'
    )

    # Load ratings data
    ratings = pd.read_table(
        os.path.join(data_path, 'ratings.dat'),
        sep='::',
        header=None,
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )

    # Load movies data
    movies = pd.read_table(
        os.path.join(data_path, 'movies.dat'),
        sep='::',
        header=None,
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='ISO-8859-1'  # Handle special characters in movie titles
    )

    return users, ratings, movies


def create_ratings_matrix(ratings: pd.DataFrame, movies: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a user-movie ratings matrix and return movie indices.

    Args:
        ratings: DataFrame containing user ratings
        movies: DataFrame containing movie information

    Returns:
        Tuple of (ratings_matrix, movie_indices)
    """
    # Create pivot table: users as rows, movies as columns
    ratings_matrix = ratings.pivot_table(
        values='rating',
        index='user_id',
        columns='movie_id',
        fill_value=0
    )

    # Create a mapping between movie IDs and their titles
    movie_mapping = dict(zip(movies['movie_id'], movies['title']))

    # Rename columns to movie titles for better readability
    ratings_matrix.columns = [movie_mapping.get(movie_id, f"Movie {movie_id}") for movie_id in ratings_matrix.columns]

    # Store movie indices for reference
    movie_indices = ratings_matrix.columns

    return ratings_matrix, movie_indices


def extract_movie_features(movies: pd.DataFrame) -> Dict[int, list]:
    """
    Extract features from movies (genres) for content-based filtering.

    Args:
        movies: DataFrame containing movie information

    Returns:
        Dictionary mapping movie_id to genre features
    """
    # Extract all unique genres
    all_genres = set()
    for genres in movies['genres'].str.split('|'):
        all_genres.update(genres)

    # Create genre features for each movie
    movie_features = {}
    for _, row in movies.iterrows():
        movie_id = row['movie_id']
        movie_genres = set(row['genres'].split('|'))
        # Create a binary feature vector for genres
        genre_vector = [1 if genre in movie_genres else 0 for genre in all_genres]
        movie_features[movie_id] = genre_vector

    return movie_features