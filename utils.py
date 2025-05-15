import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import re


def extract_year_from_title(title: str) -> Tuple[str, int]:
    """
    Extract the year from a movie title and clean the title.

    Args:
        title: Movie title with year in parentheses

    Returns:
        Tuple of (clean_title, year)
    """
    year_pattern = r"\((\d{4})\)$"
    match = re.search(year_pattern, title)

    if match:
        year = int(match.group(1))
        clean_title = title[:match.start()].strip()
        return clean_title, year

    return title, None


def get_genre_distribution(movies: pd.DataFrame) -> Dict[str, int]:
    """
    Get the distribution of genres in the dataset.

    Args:
        movies: DataFrame containing movie information

    Returns:
        Dictionary mapping genre to count
    """
    genre_counts = {}

    for genres in movies['genres']:
        for genre in genres.split('|'):
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1

    return dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))


def plot_genre_distribution(genre_counts: Dict[str, int], top_n: int = 10):
    """
    Plot the distribution of the top N genres.

    Args:
        genre_counts: Dictionary mapping genre to count
        top_n: Number of top genres to show
    """
    top_genres = dict(list(genre_counts.items())[:top_n])

    plt.figure(figsize=(12, 6))
    plt.bar(top_genres.keys(), top_genres.values())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    plt.title(f'Distribution of Top {top_n} Movie Genres')
    plt.tight_layout()
    return plt


def get_decade_distribution(movies: pd.DataFrame) -> Dict[str, int]:
    """
    Get the distribution of movies by decade.

    Args:
        movies: DataFrame containing movie information

    Returns:
        Dictionary mapping decade to count
    """
    decades = {}

    for title in movies['title']:
        _, year = extract_year_from_title(title)
        if year:
            decade = f"{(year // 10) * 10}s"
            if decade in decades:
                decades[decade] += 1
            else:
                decades[decade] = 1

    return dict(sorted(decades.items()))


def create_user_profile(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame) -> Dict[str, object]:
    """
    Create a profile for a user with their rating statistics.

    Args:
        user_id: ID of the user
        ratings: DataFrame containing user ratings
        movies: DataFrame containing movie information

    Returns:
        Dictionary containing user profile information
    """
    user_ratings = ratings[ratings['user_id'] == user_id]

    if user_ratings.empty:
        return None

    # Get basic statistics
    profile = {
        'user_id': user_id,
        'num_ratings': len(user_ratings),
        'avg_rating': user_ratings['rating'].mean(),
        'min_rating': user_ratings['rating'].min(),
        'max_rating': user_ratings['rating'].max(),
    }

    # Get top rated movies
    top_movies = user_ratings.sort_values('rating', ascending=False).head(5)
    top_movies = pd.merge(top_movies, movies, on='movie_id')
    profile['top_rated_movies'] = list(zip(top_movies['title'], top_movies['rating']))

    # Get genre preferences
    user_movie_ids = user_ratings['movie_id'].tolist()
    user_movies = movies[movies['movie_id'].isin(user_movie_ids)]

    genre_counts = {}
    for genres in user_movies['genres']:
        for genre in genres.split('|'):
            if genre in genre_counts:
                genre_counts[genre] += 1
            else:
                genre_counts[genre] = 1

    profile['favorite_genres'] = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    return profile