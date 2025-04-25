import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import current_app
import scipy.sparse as sp
import re

class ContentBasedRecommender:
    """
    Content-based recommender using movie attributes like genres, tags, etc.
    """
    def __init__(self):
        self.movies_df = None
        self.tags_df = None
        self.content_matrix = None
        self.similarity_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.initialized = False
    
    def initialize(self):
        """Load data and prepare the model"""
        if self.initialized:
            return
        
        # Load movie and tag data
        movies_path = os.path.join(current_app.config['MOVIE_DATA_PATH'], 'movies.csv')
        tags_path = os.path.join(current_app.config['MOVIE_DATA_PATH'], 'tags.csv')
        
        self.movies_df = pd.read_csv(movies_path)
        
        # Check if tags file exists (it's optional)
        if os.path.exists(tags_path):
            self.tags_df = pd.read_csv(tags_path)
        else:
            self.tags_df = None
        
        # Create content features
        self._create_content_features()
        
        # Compute similarity matrix
        similarity_matrix_path = current_app.config['CONTENT_SIMILARITY_MATRIX_PATH']
        if os.path.exists(similarity_matrix_path):
            # Load pre-computed similarity matrix
            self.similarity_matrix = sp.load_npz(similarity_matrix_path)
        else:
            # Compute similarity matrix
            self._compute_content_similarity()
            
            # Save the similarity matrix
            os.makedirs(os.path.dirname(similarity_matrix_path), exist_ok=True)
            sp.save_npz(similarity_matrix_path, self.similarity_matrix)
        
        self.initialized = True
    
    def _extract_year_from_title(self, title):
        """Extract year from movie title if present"""
        year_match = re.search(r'\((\d{4})\)$', title)
        if year_match:
            return int(year_match.group(1))
        return None
    
    def _create_content_features(self):
        """Create content features from movie attributes"""
        # Create mappings between movie IDs and matrix indices
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(self.movies_df['movieId'])}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
        # Extract year from title if available
        self.movies_df['year'] = self.movies_df['title'].apply(self._extract_year_from_title)
        
        # Prepare content features
        # Start with genres
        self.movies_df['content_features'] = self.movies_df['genres'].str.replace('|', ' ')
        
        # Add tags if available
        if self.tags_df is not None:
            # Group tags by movie
            movie_tags = self.tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
            
            # Merge tags into movies dataframe
            self.movies_df = self.movies_df.merge(movie_tags, on='movieId', how='left')
            
            # Combine genres and tags
            self.movies_df['content_features'] = self.movies_df.apply(
                lambda row: f"{row['content_features']} {row.get('tag', '')}", 
                axis=1
            )
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(
            stop_words='english',
            min_df=3,
            max_features=5000,
            analyzer='word'
        )
        
        self.content_matrix = tfidf.fit_transform(self.movies_df['content_features'].fillna(''))
        
    def _compute_content_similarity(self):
        """Compute content-based similarity matrix"""
        # Compute cosine similarity between movies
        self.similarity_matrix = cosine_similarity(self.content_matrix, dense_output=False)
    
    def get_similar_movies(self, movie_id, num_recommendations=10):
        """
        Get similar movies to a given movie based on content similarity
        
        Args:
            movie_id: ID of the movie to find similar movies for
            num_recommendations: Number of similar movies to return
            
        Returns:
            similar_movies: List of similar movie dictionaries
            explanations: List of explanation strings
        """
        # Initialize if needed
        if not self.initialized:
            self.initialize()
        
        # Check if movie exists in our dataset
        if movie_id not in self.movie_id_to_idx:
            return [], []
        
        # Get the movie index
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # Get similarity scores
        similarity_scores = self.similarity_matrix[movie_idx].toarray().flatten()
        
        # Get top similar movie indices (excluding the movie itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:num_recommendations+1]
        
        # Convert to movie information
        similar_movies = []
        explanations = []
        
        # Get source movie data for explanation
        source_movie = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
        source_genres = source_movie['genres'].split('|')
        
        for idx in similar_indices:
            movie_id = self.idx_to_movie_id[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0].to_dict()
            
            # Add similarity score
            similarity = float(similarity_scores[idx])
            movie_info['similarityScore'] = round(similarity, 2)
            
            # Parse genres into a list
            movie_info['genres'] = movie_info['genres'].split('|')
            
            # Generate explanation
            movie_genres = movie_info['genres']
            common_genres = set(source_genres).intersection(set(movie_genres))
            
            if common_genres:
                genres_text = ", ".join(common_genres)
                explanation = f"Similar to {source_movie['title']} in genres: {genres_text}"
            else:
                explanation = f"Similar to {source_movie['title']} based on content analysis"
            
            similar_movies.append(movie_info)
            explanations.append(explanation)
        
        return similar_movies, explanations
    
    def recommend(self, movie_ids, num_recommendations=10):
        """
        Generate recommendations based on a list of movie IDs the user likes
        
        Args:
            movie_ids: List of movie IDs the user likes
            num_recommendations: Number of recommendations to return
            
        Returns:
            recommendations: List of recommended movie dictionaries
            explanations: List of explanation strings
        """
        # Initialize if needed
        if not self.initialized:
            self.initialize()
        
        # Check if we have any valid movie IDs
        valid_movie_ids = [mid for mid in movie_ids if mid in self.movie_id_to_idx]
        
        if not valid_movie_ids:
            return [], []
        
        # Get movie indices
        movie_indices = [self.movie_id_to_idx[mid] for mid in valid_movie_ids]
        
        # Calculate average similarity for each movie in the dataset
        combined_scores = np.zeros(len(self.movie_id_to_idx))
        
        for idx in movie_indices:
            # Get similarity scores for this movie
            similarity_scores = self.similarity_matrix[idx].toarray().flatten()
            combined_scores += similarity_scores
        
        # Average the scores
        combined_scores /= len(movie_indices)
        
        # Set scores for input movies to 0 to exclude them
        for idx in movie_indices:
            combined_scores[idx] = 0
        
        # Get top movie indices
        top_indices = np.argsort(combined_scores)[::-1][:num_recommendations]
        
        # Convert to movie information
        recommendations = []
        explanations = []
        
        for idx in top_indices:
            movie_id = self.idx_to_movie_id[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0].to_dict()
            
            # Add relevance score
            movie_info['relevanceScore'] = round(float(combined_scores[idx]), 2)
            
            # Parse genres into a list
            movie_info['genres'] = movie_info['genres'].split('|')
            
            # Generate explanation
            # Find which input movies contributed most to this recommendation
            contributing_movies = []
            for input_idx in movie_indices:
                input_movie_id = self.idx_to_movie_id[input_idx]
                input_movie = self.movies_df[self.movies_df['movieId'] == input_movie_id].iloc[0]
                
                # Get similarity between this input movie and the recommendation
                similarity = self.similarity_matrix[input_idx, idx]
                
                if similarity > 0.1:  # Only include significant contributors
                    contributing_movies.append({
                        'title': input_movie['title'],
                        'similarity': similarity
                    })
            
            # Sort by similarity and take top 2
            contributing_movies = sorted(contributing_movies, key=lambda x: x['similarity'], reverse=True)[:2]
            
            # Generate explanation text
            if contributing_movies:
                movie_mentions = ", ".join([m['title'] for m in contributing_movies])
                explanation = f"Recommended because you liked {movie_mentions}. Similar content features including genres."
            else:
                explanation = "Recommended based on the genres and content of movies you like."
            
            recommendations.append(movie_info)
            explanations.append(explanation)
        
        return recommendations, explanations