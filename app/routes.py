from flask import Blueprint, request, jsonify, current_app, g
import pandas as pd
import os
import json
from .recommenders.collaborative import CollaborativeRecommender
from .recommenders.content import ContentBasedRecommender
from .recommenders.hybrid import HybridRecommender

bp = Blueprint('api', __name__, url_prefix='/api/v1')

def get_recommenders():
    """Get or initialize recommender instances"""
    if 'recommenders' not in g:
        g.recommenders = {
            'collaborative': CollaborativeRecommender(),
            'content': ContentBasedRecommender(),
            'hybrid': HybridRecommender()
        }
    return g.recommenders

@bp.route('/movies', methods=['GET'])
def get_movies():
    """Get a list of movies, with optional pagination and search"""
    page = request.args.get('page', default=1, type=int)
    limit = min(request.args.get('limit', default=50, type=int), 100)
    search = request.args.get('search', default='', type=str)
    
    # Load movies dataset
    movies_path = os.path.join(current_app.config['MOVIE_DATA_PATH'], 'movies.csv')
    movies_df = pd.read_csv(movies_path)
    
    # Apply search filter if provided
    if search:
        movies_df = movies_df[movies_df['title'].str.contains(search, case=False)]
    
    # Calculate pagination
    total = len(movies_df)
    start_idx = (page - 1) * limit
    end_idx = min(start_idx + limit, total)
    
    # Get paginated results
    paginated_movies = movies_df.iloc[start_idx:end_idx]
    
    # Convert to list of dictionaries
    movies_list = paginated_movies.to_dict('records')
    
    return jsonify({
        'movies': movies_list,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total,
            'pages': (total + limit - 1) // limit
        }
    })

@bp.route('/movies/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """Get details for a specific movie"""
    # Load movies dataset
    movies_path = os.path.join(current_app.config['MOVIE_DATA_PATH'], 'movies.csv')
    movies_df = pd.read_csv(movies_path)
    
    # Find the movie
    movie = movies_df[movies_df['movieId'] == movie_id]
    
    if movie.empty:
        return jsonify({'error': 'Movie not found'}), 404
    
    # Convert to dictionary
    movie_dict = movie.iloc[0].to_dict()
    
    # Load ratings data to calculate average rating
    ratings_path = os.path.join(current_app.config['MOVIE_DATA_PATH'], 'ratings.csv')
    ratings_df = pd.read_csv(ratings_path)
    
    movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]
    if not movie_ratings.empty:
        avg_rating = movie_ratings['rating'].mean()
        num_ratings = len(movie_ratings)
    else:
        avg_rating = 0
        num_ratings = 0
    
    movie_dict['averageRating'] = round(avg_rating, 1)
    movie_dict['numRatings'] = num_ratings
    
    # Parse genres into a list
    movie_dict['genres'] = movie_dict['genres'].split('|')
    
    return jsonify(movie_dict)

@bp.route('/recommendations/collaborative', methods=['POST'])
def collaborative_recommendations():
    """Get collaborative filtering recommendations based on user ratings"""
    data = request.get_json()
    
    if not data or 'userRatings' not in data:
        return jsonify({'error': 'User ratings are required'}), 400
    
    user_ratings = data['userRatings']
    count = min(data.get('count', 10), current_app.config['MAX_RECOMMENDATIONS_COUNT'])
    exclude_rated = data.get('excludeRated', True)
    
    # Get recommender
    recommenders = get_recommenders()
    cf_recommender = recommenders['collaborative']
    
    # Get recommendations
    recommendations, explanations = cf_recommender.recommend(
        user_ratings=user_ratings,
        num_recommendations=count,
        exclude_rated=exclude_rated
    )
    
    return jsonify({
        'recommendations': recommendations,
        'explanations': explanations
    })

@bp.route('/recommendations/content', methods=['POST'])
def content_recommendations():
    """Get content-based recommendations based on liked movies"""
    data = request.get_json()
    
    if not data or 'movieIds' not in data:
        return jsonify({'error': 'Movie IDs are required'}), 400
    
    movie_ids = data['movieIds']
    count = min(data.get('count', 10), current_app.config['MAX_RECOMMENDATIONS_COUNT'])
    
    # Get recommender
    recommenders = get_recommenders()
    content_recommender = recommenders['content']
    
    # Get recommendations
    recommendations, explanations = content_recommender.recommend(
        movie_ids=movie_ids,
        num_recommendations=count
    )
    
    return jsonify({
        'recommendations': recommendations,
        'explanations': explanations
    })

@bp.route('/recommendations/hybrid', methods=['POST'])
def hybrid_recommendations():
    """Get hybrid recommendations based on user ratings and liked movies"""
    data = request.get_json()
    
    if not data or 'userRatings' not in data:
        return jsonify({'error': 'User ratings are required'}), 400
    
    user_ratings = data['userRatings']
    count = min(data.get('count', 10), current_app.config['MAX_RECOMMENDATIONS_COUNT'])
    exclude_rated = data.get('excludeRated', True)
    
    # Get recommender
    recommenders = get_recommenders()
    hybrid_recommender = recommenders['hybrid']
    
    # Get recommendations
    recommendations, explanations = hybrid_recommender.recommend(
        user_ratings=user_ratings,
        num_recommendations=count,
        exclude_rated=exclude_rated
    )
    
    return jsonify({
        'recommendations': recommendations,
        'explanations': explanations
    })

@bp.route('/movies/<int:movie_id>/similar', methods=['GET'])
def similar_movies(movie_id):
    """Get similar movies based on content"""
    count = min(request.args.get('count', default=10, type=int), 
                current_app.config['MAX_RECOMMENDATIONS_COUNT'])
    
    # Get recommender
    recommenders = get_recommenders()
    content_recommender = recommenders['content']
    
    # Get similar movies
    similar, explanations = content_recommender.get_similar_movies(
        movie_id=movie_id,
        num_recommendations=count
    )
    
    return jsonify({
        'similarMovies': similar,
        'explanations': explanations
    })

@bp.route('/users/<int:user_id>/recommendations', methods=['GET'])
def user_recommendations(user_id):
    """Get recommendations for an existing user in the dataset"""
    count = min(request.args.get('count', default=10, type=int), 
                current_app.config['MAX_RECOMMENDATIONS_COUNT'])
    method = request.args.get('method', default='hybrid', type=str)
    
    # Validate method
    if method not in ['collaborative', 'content', 'hybrid']:
        return jsonify({'error': 'Invalid recommendation method'}), 400
    
    # Get recommender
    recommenders = get_recommenders()
    recommender = recommenders[method]
    
    # Get user ratings
    ratings_path = os.path.join(current_app.config['MOVIE_DATA_PATH'], 'ratings.csv')
    ratings_df = pd.read_csv(ratings_path)
    
    user_ratings_df = ratings_df[ratings_df['userId'] == user_id]
    
    if user_ratings_df.empty:
        return jsonify({'error': 'User not found'}), 404
    
    # Convert to dictionary format expected by recommenders
    user_ratings = user_ratings_df[['movieId', 'rating']].to_dict('records')
    
    # Get recommendations
    recommendations, explanations = recommender.recommend(
        user_ratings=user_ratings,
        num_recommendations=count,
        exclude_rated=True
    )
    
    return jsonify({
        'recommendations': recommendations,
        'explanations': explanations
    })