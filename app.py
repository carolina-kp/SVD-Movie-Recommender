import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from typing import Dict, List, Tuple

# Import our custom modules
from src.data_loader import load_movielens_data, create_ratings_matrix
from src.recommender import MovieRecommender
from src.utils import (
    get_genre_distribution,
    plot_genre_distribution,
    get_decade_distribution,
    create_user_profile,
    extract_year_from_title
)

# Page configuration
st.set_page_config(
    page_title="SVD Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# App title
st.title("🎬 SVD Movie Recommendation System")
st.markdown("### Dimensionality Reduction with Singular Value Decomposition")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Movie Recommendations", "Movie Explorer", "User Analysis", "About SVD"]
)


# Check if data exists
@st.cache_data
def load_data():
    try:
        users, ratings, movies = load_movielens_data()
        ratings_matrix, movie_indices = create_ratings_matrix(ratings, movies)
        return users, ratings, movies, ratings_matrix, movie_indices
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None


# Load data
users, ratings, movies, ratings_matrix, movie_indices = load_data()

# Check if data was loaded successfully
if users is None or movies is None or ratings is None:
    st.error("Failed to load MovieLens data. Please check the data directory.")
    st.info("You can download the MovieLens 1M dataset from: http://files.grouplens.org/datasets/movielens/ml-1m.zip")
    st.stop()


# Initialize SVD recommender
@st.cache_resource
def get_recommender(n_components=50):
    recommender = MovieRecommender(n_components=n_components)
    recommender.fit(ratings_matrix)
    return recommender


# Set a default value for n_components
if "n_components" not in st.session_state:
    st.session_state.n_components = 50

# Home page
if page == "Home":
    st.header("Welcome to the SVD Movie Recommender")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This application uses Singular Value Decomposition (SVD) to recommend movies based on the MovieLens 1M dataset. 

        ### Dataset Statistics:
        """)

        stats_col1, stats_col2, stats_col3 = st.columns(3)

        with stats_col1:
            st.metric("Number of Users", f"{len(users):,}")

        with stats_col2:
            st.metric("Number of Movies", f"{len(movies):,}")

        with stats_col3:
            st.metric("Number of Ratings", f"{len(ratings):,}")

        # Genre distribution
        st.subheader("Genre Distribution")
        genre_counts = get_genre_distribution(movies)
        fig = px.bar(
            x=list(genre_counts.keys()),
            y=list(genre_counts.values()),
            labels={'x': 'Genre', 'y': 'Number of Movies'},
            title='Distribution of Movie Genres'
        )
        st.plotly_chart(fig)

        # Decade distribution
        st.subheader("Movies by Decade")
        decade_counts = get_decade_distribution(movies)
        fig = px.line(
            x=list(decade_counts.keys()),
            y=list(decade_counts.values()),
            markers=True,
            labels={'x': 'Decade', 'y': 'Number of Movies'},
            title='Distribution of Movies by Decade'
        )
        st.plotly_chart(fig)

    with col2:
        st.subheader("SVD Components")
        n_components = st.slider(
            "Number of SVD Components",
            min_value=10,
            max_value=100,
            value=st.session_state.n_components,
            step=5,
            key="home_n_components"
        )

        # Update session state
        st.session_state.n_components = n_components

        if st.button("Apply SVD"):
            with st.spinner("Computing SVD components..."):
                recommender = get_recommender(n_components)
                st.success(f"Explained variance: {recommender.explained_variance:.3f}")

        st.markdown("""
        ### What you can do:
        - Get movie recommendations based on a movie you like
        - Explore the movie database
        - Analyze user preferences
        - Learn about SVD and collaborative filtering

        Navigate using the sidebar!
        """)

# Movie Recommendations page
elif page == "Movie Recommendations":
    st.header("Get Movie Recommendations")

    # Get the recommender
    recommender = get_recommender(st.session_state.n_components)

    recommendation_type = st.radio(
        "Choose recommendation type:",
        ["Similar Movies", "Personalized Recommendations"]
    )

    if recommendation_type == "Similar Movies":
        st.subheader("Find Movies Similar to...")

        # Get list of movies
        movie_list = sorted(recommender.movie_indices)

        # Create a search box for movies
        search_term = st.text_input("Search for a movie:")
        if search_term:
            filtered_movies = [movie for movie in movie_list if search_term.lower() in movie.lower()]
            if not filtered_movies:
                st.warning("No movies found matching your search.")
            else:
                selected_movie = st.selectbox("Select a movie:", filtered_movies)
                st.write(f"You selected: **{selected_movie}**")

                col1, col2 = st.columns([1, 1])

                with col1:
                    # Parameters for similarity
                    min_correlation = st.slider(
                        "Minimum similarity threshold",
                        min_value=0.70,
                        max_value=0.99,
                        value=0.85,
                        step=0.01
                    )

                    max_recommendations = st.slider(
                        "Number of recommendations",
                        min_value=1,
                        max_value=20,
                        value=10
                    )

                    # Get recommendations
                    if st.button("Get Recommendations"):
                        with st.spinner("Finding similar movies..."):
                            similar_movies = recommender.get_similar_movies(
                                selected_movie,
                                min_correlation=min_correlation,
                                max_recommendations=max_recommendations
                            )

                            if similar_movies:
                                st.subheader("Recommended Movies:")
                                for i, (movie, correlation) in enumerate(similar_movies, 1):
                                    st.write(f"{i}. **{movie}** (Similarity: {correlation:.3f})")
                            else:
                                st.info(
                                    "No similar movies found with the current threshold. Try lowering the similarity threshold.")

                with col2:
                    # Visualize SVD components
                    st.subheader("SVD Components")
                    fig = recommender.visualize_svd_components(selected_movie)
                    st.pyplot(fig)
        else:
            st.info("Enter a movie title to search.")

    else:  # Personalized Recommendations
        st.subheader("Rate Movies to Get Personalized Recommendations")

        # Get a sample of popular movies for the user to rate
        popular_movies = ratings.groupby('movie_id')['rating'].count().nlargest(50).index.tolist()
        popular_movie_titles = movies[movies['movie_id'].isin(popular_movies)]['title'].tolist()

        # Randomly select 10 movies for the user to rate
        if 'movies_to_rate' not in st.session_state:
            st.session_state.movies_to_rate = np.random.choice(popular_movie_titles, 10, replace=False).tolist()

        # Store user ratings
        if 'user_ratings' not in st.session_state:
            st.session_state.user_ratings = {}

        # Let the user rate movies
        st.write("Rate at least 3 movies to get personalized recommendations:")

        # Create two columns of movies to rate
        col1, col2 = st.columns(2)

        for i, movie in enumerate(st.session_state.movies_to_rate):
            if i % 2 == 0:
                with col1:
                    rating = st.slider(
                        f"{movie}",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.0,
                        step=0.5,
                        key=f"rate_{i}"
                    )
                    if rating > 0:
                        st.session_state.user_ratings[movie] = rating
            else:
                with col2:
                    rating = st.slider(
                        f"{movie}",
                        min_value=0.0,
                        max_value=5.0,
                        value=0.0,
                        step=0.5,
                        key=f"rate_{i}"
                    )
                    if rating > 0:
                        st.session_state.user_ratings[movie] = rating

        # Button to get recommendations
        if st.button("Get Personalized Recommendations"):
            if len(st.session_state.user_ratings) < 3:
                st.warning("Please rate at least 3 movies to get recommendations.")
            else:
                with st.spinner("Generating recommendations..."):
                    recommendations = recommender.recommend_for_user(
                        st.session_state.user_ratings,
                        n_recommendations=10
                    )

                    st.subheader("Recommended Movies for You:")
                    for i, (movie, rating) in enumerate(recommendations, 1):
                        st.write(f"{i}. **{movie}** (Predicted Rating: {rating:.2f}/5.0)")

                    # Option to reset
                    if st.button("Reset Ratings"):
                        st.session_state.user_ratings = {}
                        st.session_state.movies_to_rate = np.random.choice(popular_movie_titles, 10,
                                                                           replace=False).tolist()
                        st.experimental_rerun()

# Movie Explorer page
elif page == "Movie Explorer":
    st.header("Movie Database Explorer")

    # Add search and filter options
    st.subheader("Search and Filter")

    col1, col2 = st.columns(2)

    with col1:
        search_title = st.text_input("Search by title:")

    with col2:
        # Extract unique genres
        all_genres = set()
        for genres in movies['genres']:
            all_genres.update(genres.split('|'))

        selected_genre = st.selectbox("Filter by genre:", ["All"] + sorted(all_genres))

    # Filter by decade
    decades = ["All"] + sorted([f"{(year // 10) * 10}s" for year in range(1900, 2020, 10)])
    selected_decade = st.selectbox("Filter by decade:", decades)

    # Apply filters
    filtered_movies = movies.copy()

    if search_title:
        filtered_movies = filtered_movies[filtered_movies['title'].str.contains(search_title, case=False)]

    if selected_genre != "All":
        filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(selected_genre)]

    if selected_decade != "All":
        decade_start = int(selected_decade[:-1])
        decade_pattern = f"\\({decade_start}\\d\\)"
        filtered_movies = filtered_movies[filtered_movies['title'].str.contains(decade_pattern, regex=True)]

    # Show results
    st.subheader(f"Results: {len(filtered_movies)} movies found")

    if not filtered_movies.empty:
        # Pagination
        items_per_page = 20
        num_pages = (len(filtered_movies) - 1) // items_per_page + 1

        page_num = st.number_input(
            f"Page (1-{num_pages}):",
            min_value=1,
            max_value=num_pages,
            value=1
        )

        start_idx = (page_num - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_movies))

        # Display movies
        for _, row in filtered_movies.iloc[start_idx:end_idx].iterrows():
            with st.expander(f"{row['title']}"):
                st.write(f"**Movie ID:** {row['movie_id']}")
                st.write(f"**Genres:** {row['genres'].replace('|', ', ')}")

                # Get average rating
                avg_rating = ratings[ratings['movie_id'] == row['movie_id']]['rating'].mean()
                st.write(f"**Average Rating:** {avg_rating:.2f}/5.0")

                # Number of ratings
                num_ratings = len(ratings[ratings['movie_id'] == row['movie_id']])
                st.write(f"**Number of Ratings:** {num_ratings}")
    else:
        st.info("No movies found matching your criteria.")

# User Analysis page
elif page == "User Analysis":
    st.header("User Analysis")

    # Select a user
    user_id = st.number_input("Enter User ID:", min_value=1, max_value=users['user_id'].max(), value=1)

    if st.button("Analyze User"):
        with st.spinner("Analyzing user preferences..."):
            # Get user profile
            user_profile = create_user_profile(user_id, ratings, movies)

            if user_profile:
                st.subheader(f"User Profile (ID: {user_id})")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Number of Ratings", user_profile['num_ratings'])
                    st.metric("Average Rating", f"{user_profile['avg_rating']:.2f}/5.0")

                with col2:
                    st.metric("Lowest Rating", user_profile['min_rating'])
                    st.metric("Highest Rating", user_profile['max_rating'])

                # Top rated movies
                st.subheader("Top Rated Movies")
                for movie, rating in user_profile['top_rated_movies']:
                    st.write(f"**{movie}** - {rating}/5.0")

                # Favorite genres
                st.subheader("Favorite Genres")
                genre_names = [genre for genre, count in user_profile['favorite_genres']]
                genre_counts = [count for genre, count in user_profile['favorite_genres']]

                fig = px.pie(
                    values=genre_counts,
                    names=genre_names,
                    title="Genre Preferences"
                )
                st.plotly_chart(fig)

                # Ratings distribution
                user_ratings = ratings[ratings['user_id'] == user_id]

                fig = px.histogram(
                    user_ratings,
                    x="rating",
                    nbins=10,
                    title="Rating Distribution",
                    labels={"rating": "Rating", "count": "Number of Movies"}
                )
                st.plotly_chart(fig)
            else:
                st.error(f"User with ID {user_id} not found.")

# About SVD page
elif page == "About SVD":
    st.header("About SVD and Collaborative Filtering")

    # About SVD page (continued)
    st.markdown("""
        In the context of movie recommendations:
        - The original data is a user-movie ratings matrix, which is often very sparse
        - SVD decomposes this matrix into three components: U, Σ, and V^T
            - U: User features in latent space
            - Σ: Diagonal matrix of singular values
            - V^T: Movie features in latent space

        By selecting only the top k singular values (and corresponding vectors), we can approximate the original matrix with a lower-dimensional representation, which captures the most important patterns.

        ### Collaborative Filtering

        Collaborative filtering is a technique used in recommendation systems that makes predictions about a user's interests by collecting preferences from many users. There are two main approaches:

        1. **User-Based Collaborative Filtering**: Recommends items based on similar users' preferences
            - "Users who are similar to you also liked these items"

        2. **Item-Based Collaborative Filtering**: Recommends items similar to those the user has liked before
            - "Because you liked this item, you might also like these similar items"

        Our implementation uses SVD to enable a form of model-based collaborative filtering, learning latent factors that describe both users and items.

        ### Advantages of SVD for Recommendations

        - **Dimensionality Reduction**: Reduces the feature space to manageable dimensions
        - **Handles Sparsity**: Works well with sparse data, which is common in recommendation systems
        - **Discovers Latent Factors**: Identifies hidden patterns that may not be obvious
        - **Scalability**: Once the model is trained, recommendations can be generated quickly

        ### Limitations

        - **Cold Start Problem**: Difficulty in recommending for new users or items
        - **Computational Complexity**: Training can be computationally expensive for large datasets
        - **Dynamic Updates**: Difficult to update the model incrementally as new data arrives
        """)

    # Explain the implementation
    st.subheader("Our Implementation")

    st.markdown("""
        In our implementation, we use scikit-learn's TruncatedSVD, which is particularly well-suited for sparse matrices. The key components are:

        1. **Data Preparation**: Convert the user-movie ratings into a matrix format
        2. **SVD Computation**: Apply TruncatedSVD to reduce the dimensionality
        3. **Similarity Calculation**: Compute correlations between movies in the reduced space
        4. **Recommendation Generation**: Use the similarity measures to recommend movies

        The slider on the home page allows you to adjust the number of SVD components, which controls the trade-off between model complexity and accuracy. More components can capture more detail but might also include noise.
        """)

    # Show a diagram of SVD
    st.subheader("SVD Visualization")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Example visualization of the SVD concept
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/SVD-Numpy.png/800px-SVD-Numpy.png",
                 caption="SVD Decomposition",
                 use_column_width=True)

    with col2:
        st.markdown("""
            The diagram shows how a matrix is decomposed using SVD into:

            - U (left singular vectors)
            - Σ (singular values)
            - V^T (right singular vectors)

            By keeping only the top k singular values, we can create a low-rank approximation of the original matrix.
            """)

    # Project credits
    st.subheader("Project Credits")

    st.markdown("""
        This application is based on the material from:

        - Machine Learning for Business Analytics & AI course
        - Prof. Dr. Javier Panadero (UAB)
        - Prof. Dr. Angel A. Juan (UPV)

        The original code has been extended and improved to create this interactive application.
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### SVD Parameters")
sidebar_n_components = st.sidebar.slider(
    "Number of SVD components",
    min_value=10,
    max_value=100,
    value=st.session_state.n_components,
    step=5,
    key="sidebar_n_components"
)

# Update the session state if changed from sidebar
if sidebar_n_components != st.session_state.n_components:
    st.session_state.n_components = sidebar_n_components
    st.experimental_rerun()

# Add a footer with project info
st.sidebar.markdown("---")
st.sidebar.info(
    "Created for the Dimensionality Reduction Apps project. "
    "Based on the MovieLens 1M dataset."
)