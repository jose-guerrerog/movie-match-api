import numpy as np
from flask import current_app
from .collaborative import CollaborativeRecommender
from .content import ContentBasedRecommender

class HybridRecommender:
    """
    Hybrid recommender combining collaborative filtering and content-based approaches
    """
    def __init__(self):
        self.collaborative_recommender = CollaborativeRecommender()
        self.content_based_recommender = ContentBasedRecommender()
        self.initialized = False
    
    def initialize(self):
        """Initialize component recommenders"""
        if not self.initialized:
            self.collaborative_recommender.initialize()
            self.content_based_recommender.initialize()
            self.initialized = True
    
    def recommend(self, user_ratings, num_recommendations=10, exclude_rated=True):
        """
        Generate hybrid recommendations using both collaborative and content-based approaches
        
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
        
        # Get collaborative filtering recommendations
        cf_recommendations, cf_explanations = self.collaborative_recommender.recommend(
            user_ratings=user_ratings,
            num_recommendations=num_recommendations * 2,  # Get more to merge later
            exclude_rated=exclude_rated
        )
        
        # Extract movie IDs from user ratings with high scores (>= 4) for content-based filtering
        liked_movie_ids = [item['movieId'] for item in user_ratings if item['rating'] >= 4.0]
        
        # If user hasn't rated any movies highly, use all rated movies
        if not liked_movie_ids:
            liked_movie_ids = [item['movieId'] for item in user_ratings]
        
        # Get content-based recommendations
        cb_recommendations, cb_explanations = self.content_based_recommender.recommend(
            movie_ids=liked_movie_ids,
            num_recommendations=num_recommendations * 2  # Get more to merge later
        )
        
        # Get weights from config
        cf_weight = current_app.config.get('COLLABORATIVE_WEIGHT', 0.6)
        cb_weight = current_app.config.get('CONTENT_WEIGHT', 0.4)
        
        # Normalize weights
        total_weight = cf_weight + cb_weight
        cf_weight /= total_weight
        cb_weight /= total_weight
        
        # Create a dictionary to store combined scores and explanations
        all_recommendations = {}
        
        # Process collaborative filtering recommendations
        for i, rec in enumerate(cf_recommendations):
            movie_id = rec['movieId']
            
            if movie_id not in all_recommendations:
                all_recommendations[movie_id] = {
                    'movie_info': rec,
                    'cf_score': rec.get('predictedRating', 0) / 5.0,  # Normalize to 0-1
                    'cb_score': 0,
                    'cf_explanation': cf_explanations[i],
                    'cb_explanation': None
                }
        
        # Process content-based recommendations
        for i, rec in enumerate(cb_recommendations):
            movie_id = rec['movieId']
            
            if movie_id not in all_recommendations:
                all_recommendations[movie_id] = {
                    'movie_info': rec,
                    'cf_score': 0,
                    'cb_score': rec.get('relevanceScore', 0),  # Already normalized
                    'cf_explanation': None,
                    'cb_explanation': cb_explanations[i]
                }
            else:
                # Update with content-based score and explanation
                all_recommendations[movie_id]['cb_score'] = rec.get('relevanceScore', 0)
                all_recommendations[movie_id]['cb_explanation'] = cb_explanations[i]
        
        # Calculate weighted scores
        for movie_id, data in all_recommendations.items():
            data['combined_score'] = (cf_weight * data['cf_score']) + (cb_weight * data['cb_score'])
        
        # Sort by combined score and take top N
        sorted_recommendations = sorted(
            all_recommendations.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )[:num_recommendations]
        
        # Prepare final output
        recommendations = []
        explanations = []
        
        for data in sorted_recommendations:
            movie_info = data['movie_info']
            
            # Add combined score
            movie_info['combinedScore'] = round(data['combined_score'], 2)
            
            # Generate hybrid explanation
            if data['cf_explanation'] and data['cb_explanation']:
                cf_contrib = data['cf_score'] * cf_weight / data['combined_score'] if data['combined_score'] > 0 else 0
                cb_contrib = data['cb_score'] * cb_weight / data['combined_score'] if data['combined_score'] > 0 else 0
                
                # Use the explanation that contributed more to the recommendation
                if cf_contrib >= cb_contrib:
                    explanation = data['cf_explanation']
                else:
                    explanation = data['cb_explanation']
            elif data['cf_explanation']:
                explanation = data['cf_explanation']
            elif data['cb_explanation']:
                explanation = data['cb_explanation']
            else:
                explanation = "Recommended based on your rating history and movie preferences."
            
            recommendations.append(movie_info)
            explanations.append(explanation)
        
        return recommendations, explanations