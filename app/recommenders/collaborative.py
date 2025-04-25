import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import current_app
import joblib

class CollaborativeRecommender:
    """
    Collaborative filtering recommender using user-based approach
    """
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.initialized = False
    
    def initialize(self):
        """Load data and prepare the model"""
        if self.initialized:
            return
        
        # Load movie and rating data
        movies_path = os.path.join(current_app.config['MOVIE_DATA_PATH'], 'movies.csv')
        ratings_path = os.path.join(current_app.config['MOVIE_DATA_PATH'], 'ratings.csv')
        
        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Create user-item matrix
        self._create_user_item_matrix()
        
        # Load pre-computed model if available
        model_path = current_app.config['COLLABORATIVE_MODEL_PATH']
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            self.user_similarity_matrix = model_data['user_similarity']
        else:
            # Compute user similarity matrix
            self._compute_user_similarity()
            
            # Save the model
            model_data = {
                'user_similarity': self.user_similarity_matrix
            }
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model_data, model_path)
        
        self.initialized = True
    
    def _create_user_item_matrix(self):
        """Create user-item rating matrix"""
        # Create mappings between movie IDs and matrix indices
        unique_movie_ids = self.ratings_df['movieId'].unique()
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
        # Create the user-item matrix (users as rows, items as columns)
        user_ids = self.ratings_df['userId'].unique()
        n_users = len(user_ids)
        n_items = len(unique_movie_ids)
        
        # Create mapping between userIds and matrix indices
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        
        # Initialize matrix with zeros
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        # Fill the matrix with ratings
        for _, row in self.ratings_df.iterrows():
            user_idx = user_id_to_idx[row['userId']]
            movie_idx = self.movie_id_to_idx[row['movieId']]
            self.user_item_matrix[user_idx, movie_idx] = row['rating']
    
    def _compute_user_similarity(self):
        """Compute user-user similarity matrix using cosine similarity"""
        # Compute cosine similarity between users
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        
        # Set self-similarity to 0 to avoid recommending already seen items
        np.fill_diagonal(self.user_similarity_matrix, 0)
    
    def recommend(self, user_ratings, num_recommendations=10, exclude_rated=True):
        """
        Generate recommendations for a new user based on their ratings
        
        Args:
            user_ratings: List of dictionaries containing movieId and rating
            num_recommendations: Number of recommendations to return
            exclude_rated: Whether to exclude movies the user has already rated
            
        Returns:
            recommendations: List of recommended movie dictionaries
            explanations: List of explanation strings
        """
        # Initialize if needed
        if not self.initialized:
            self.initialize()
        
        # Create a user vector from the provided ratings
        user_vector = np.zeros(len(self.movie_id_to_idx))
        rated_movie_indices = []
        
        for item in user_ratings:
            movie_id = item['movieId']
            rating = item['rating']
            
            # Check if the movie is in our dataset
            if movie_id in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[movie_id]
                user_vector[idx] = rating
                rated_movie_indices.append(idx)
        
        # Check if the user has rated any movies
        if len(rated_movie_indices) == 0:
            return [], []
        
        # Compute similarity between this user and all users in the dataset
        user_similarities = cosine_similarity(
            user_vector.reshape(1, -1), 
            self.user_item_matrix
        )[0]
        
        # Get top similar users
        top_similar_users = np.argsort(user_similarities)[::-1][:100]
        
        # Calculate weighted ratings for all movies
        weighted_ratings = np.zeros(len(self.movie_id_to_idx))
        similarity_sums = np.zeros(len(self.movie_id_to_idx))
        
        for user_idx in top_similar_users:
            # Get similarity score
            similarity = user_similarities[user_idx]
            
            # Skip users with zero similarity
            if similarity <= 0:
                continue
            
            # Get user's ratings
            user_ratings_vector = self.user_item_matrix[user_idx]
            
            # Add weighted ratings
            for movie_idx, rating in enumerate(user_ratings_vector):
                if rating > 0:  # User has rated this movie
                    weighted_ratings[movie_idx] += similarity * rating
                    similarity_sums[movie_idx] += similarity
        
        # Calculate predicted ratings
        predicted_ratings = np.zeros(len(self.movie_id_to_idx))
        for i in range(len(predicted_ratings)):
            if similarity_sums[i] > 0:
                predicted_ratings[i] = weighted_ratings[i] / similarity_sums[i]
        
        # Exclude movies the user has already rated if requested
        if exclude_rated:
            for idx in rated_movie_indices:
                predicted_ratings[idx] = 0
        
        # Get top-N movie indices
        top_indices = np.argsort(predicted_ratings)[::-1][:num_recommendations]
        
        # Convert to movie information
        recommendations = []
        explanations = []
        
        for idx in top_indices:
            if predicted_ratings[idx] <= 0:
                continue
                
            movie_id = self.idx_to_movie_id[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0].to_dict()
            
            # Add predicted rating
            movie_info['predictedRating'] = round(float(predicted_ratings[idx]), 1)
            
            # Parse genres into a list
            movie_info['genres'] = movie_info['genres'].split('|')
            
            # Generate explanation
            # Find movies the user rated highly that influenced this recommendation
            explanation_movies = []
            for rated_idx in rated_movie_indices:
                rated_movie_id = self.idx_to_movie_id[rated_idx]
                rated_movie = self.movies_df[self.movies_df['movieId'] == rated_movie_id].iloc[0]
                
                # Check if there's genre overlap as a simple proxy for similarity
                rated_genres = set(rated_movie['genres'].split('|'))
                recommended_genres = set(movie_info['genres'])
                
                if len(rated_genres.intersection(recommended_genres)) > 0:
                    rating = user_vector[rated_idx]
                    if rating >= 3.5:  # Only include if rated positively
                        explanation_movies.append({
                            'title': rated_movie['title'],
                            'rating': rating
                        })
            
            # Sort by rating and take top 3
            explanation_movies = sorted(explanation_movies, key=lambda x: x['rating'], reverse=True)[:3]
            
            # Generate explanation text
            if explanation_movies:
                movie_mentions = ", ".join([f"{m['title']} ({m['rating']})" for m in explanation_movies])
                explanation = f"Recommended because you liked {movie_mentions}. Users with similar tastes enjoyed this movie."
            else:
                explanation = "Users with similar taste to yours enjoyed this movie."
            
            recommendations.append(movie_info)
            explanations.append(explanation)
        
        return recommendations, explanations