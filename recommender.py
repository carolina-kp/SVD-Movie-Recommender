import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class MovieRecommender:
    """SVD-based movie recommendation system with enhanced features"""

    def __init__(self, n_components: int = 50, random_state: int = 101):
        """
        Initialize the movie recommender.

        Args:
            n_components: Number of latent factors (SVD components)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.ratings_matrix = None
        self.movie_indices = None
        self.correlation_matrix = None
        self.R = None
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.explained_variance = 0

    def fit(self, ratings_matrix: pd.DataFrame):
        """
        Fit the SVD model on the ratings matrix.

        Args:
            ratings_matrix: User-movie ratings matrix
        """
        self.ratings_matrix = ratings_matrix
        self.movie_indices = ratings_matrix.columns

        # Create mappings between movie titles and indices
        self.movie_to_idx = {movie: i for i, movie in enumerate(self.movie_indices)}
        self.idx_to_movie = {i: movie for i, movie in enumerate(self.movie_indices)}

        # Fit SVD model
        self.R = self.svd.fit_transform(ratings_matrix.values.T)

        # Store explained variance
        self.explained_variance = np.sum(self.svd.explained_variance_ratio_)
        print(f"Explained variance by {self.n_components} components: {self.explained_variance:.3f}")

        # Compute correlation matrix for item-based recommendations
        self.correlation_matrix = np.corrcoef(self.R)

        return self

    def get_movie_svd_profile(self, movie_title: str) -> np.ndarray:
        """
        Get the SVD components for a specific movie.

        Args:
            movie_title: Title of the movie

        Returns:
            SVD component vector for the movie
        """
        if movie_title not in self.movie_to_idx:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset")

        movie_idx = self.movie_to_idx[movie_title]
        return self.R[movie_idx]

    def get_similar_movies(self, movie_title: str, min_correlation: float = 0.85, max_recommendations: int = 10) -> \
    List[Tuple[str, float]]:
        """
        Find movies similar to the given movie based on correlation.

        Args:
            movie_title: Title of the reference movie
            min_correlation: Minimum correlation threshold
            max_recommendations: Maximum number of recommendations

        Returns:
            List of (movie_title, correlation) tuples for similar movies
        """
        if movie_title not in self.movie_to_idx:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset")

        movie_idx = self.movie_to_idx[movie_title]

        # Get correlations for the specified movie
        correlations = self.correlation_matrix[movie_idx]

        # Find movies with high correlation (excluding the movie itself)
        similar_indices = np.where((correlations >= min_correlation) & (correlations < 1.0))[0]

        # Sort by correlation (highest first)
        similar_indices = sorted(similar_indices, key=lambda i: correlations[i], reverse=True)

        # Limit the number of recommendations
        similar_indices = similar_indices[:max_recommendations]

        # Return movie titles and their correlations
        return [(self.idx_to_movie[idx], correlations[idx]) for idx in similar_indices]

    def visualize_svd_components(self, movie_title: str, top_n: int = 5):
        """
        Visualize the SVD components for a movie.

        Args:
            movie_title: Title of the movie
            top_n: Number of top components to highlight
        """
        if movie_title not in self.movie_to_idx:
            raise ValueError(f"Movie '{movie_title}' not found in the dataset")

        movie_idx = self.movie_to_idx[movie_title]
        svd_components = self.R[movie_idx]

        # Find the indices of the top components
        top_component_indices = np.abs(svd_components).argsort()[-top_n:][::-1]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(svd_components)), svd_components)

        # Highlight the top components
        for idx in top_component_indices:
            bars[idx].set_color('red')

        plt.xlabel('SVD Component')
        plt.ylabel('Value')
        plt.title(f'SVD Components for "{movie_title}"')
        plt.xticks(range(0, len(svd_components), max(1, len(svd_components) // 10)))
        return plt

    def recommend_for_user(self, user_ratings: Dict[str, float], n_recommendations: int = 10) -> List[
        Tuple[str, float]]:
        """
        Recommend movies for a user based on their ratings.

        Args:
            user_ratings: Dictionary mapping movie titles to ratings
            n_recommendations: Number of recommendations to return

        Returns:
            List of (movie_title, predicted_rating) tuples
        """
        # Check if the provided movies are in the dataset
        for movie in user_ratings:
            if movie not in self.movie_to_idx:
                raise ValueError(f"Movie '{movie}' not found in the dataset")

        # Create a user vector
        user_vector = np.zeros(len(self.movie_indices))
        for movie, rating in user_ratings.items():
            user_vector[self.movie_to_idx[movie]] = rating

        # Transform user vector to latent space
        user_profile = np.dot(user_vector, self.R) / len(user_ratings)

        # Compute predicted ratings
        predicted_ratings = np.dot(user_profile, self.R.T)

        # Sort movies by predicted rating
        sorted_indices = np.argsort(predicted_ratings)[::-1]

        # Filter out movies that the user has already rated
        recommendations = []
        for idx in sorted_indices:
            movie = self.idx_to_movie[idx]
            if movie not in user_ratings:
                recommendations.append((movie, predicted_ratings[idx]))
                if len(recommendations) >= n_recommendations:
                    break

        return recommendations