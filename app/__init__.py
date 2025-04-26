from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from database import SessionLocal, engine, Base
import models
from sqlalchemy.sql import func
from sqlalchemy import desc

app = Flask(__name__)
CORS(app)

# Create all tables in the database
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def get_movie_info(movie_id):
    """Get detailed movie information from database"""
    db = get_db()
    movie = db.query(models.Movie).filter(models.Movie.movieId == movie_id).first()
    if not movie:
        return None
    
    return {
        'id': int(movie.movieId),
        'title': movie.title,
        'genres': movie.genres.split('|'),
        'year': movie.title[-5:-1] if movie.title[-5:-1].isdigit() else None,
    }

def explain_content_recommendation(movie_id, recommended_id):
    """Generate explanation for content-based recommendation"""
    db = get_db()
    movie = db.query(models.Movie).filter(models.Movie.movieId == movie_id).first()
    rec_movie = db.query(models.Movie).filter(models.Movie.movieId == recommended_id).first()
    
    # Compare genres
    movie_genres = set(movie.genres.split('|'))
    rec_genres = set(rec_movie.genres.split('|'))
    shared_genres = movie_genres.intersection(rec_genres)
    
    if shared_genres:
        return f"This movie shares similar genres with '{movie.title}': {', '.join(shared_genres)}"
    else:
        return f"This movie has similar themes to '{movie.title}'"

def explain_collaborative_recommendation(movie_id, recommended_id):
    """Generate explanation for collaborative recommendation"""
    db = get_db()
    movie = db.query(models.Movie).filter(models.Movie.movieId == movie_id).first()
    
    # Find users who rated both movies highly
    users_who_liked_movie1 = db.query(models.Rating.userId).filter(
        models.Rating.movieId == movie_id,
        models.Rating.rating >= 4
    ).all()
    users_who_liked_movie1 = set([user[0] for user in users_who_liked_movie1])
    
    users_who_liked_movie2 = db.query(models.Rating.userId).filter(
        models.Rating.movieId == recommended_id,
        models.Rating.rating >= 4
    ).all()
    users_who_liked_movie2 = set([user[0] for user in users_who_liked_movie2])
    
    users_rated_both = users_who_liked_movie1.intersection(users_who_liked_movie2)
    count = len(users_rated_both)
    
    if count > 0:
        return f"{count} users who liked '{movie.title}' also rated this movie highly"
    else:
        return f"Users with similar taste to those who enjoyed '{movie.title}' liked this movie"

@app.route('/api/recommend', methods=['GET'])
def recommend():
    movie_id = int(request.args.get('movie_id', '1'))
    count = int(request.args.get('count', '5'))
    method = request.args.get('method', 'hybrid')  # 'content', 'collaborative', or 'hybrid'
    
    try:
        db = get_db()
        
        # Check if movie exists
        movie = db.query(models.Movie).filter(models.Movie.movieId == movie_id).first()
        if not movie:
            return jsonify({"error": "Movie not found"}), 404
        
        recommendations = []
        
        if method == 'content' or method == 'hybrid':
            # Get content-based recommendations
            content_recs = db.query(
                models.ContentSimilarity.similar_movie_id,
                models.ContentSimilarity.similarity_score
            ).filter(
                models.ContentSimilarity.movie_id == movie_id
            ).order_by(
                desc(models.ContentSimilarity.similarity_score)
            ).limit(count+10).all()
            
            content_recommendations = [rec[0] for rec in content_recs if rec[0] != movie_id]
        else:
            content_recommendations = []
            
        if method == 'collaborative' or method == 'hybrid':
            # Get collaborative recommendations
            collab_recs = db.query(
                models.ItemSimilarity.similar_movie_id,
                models.ItemSimilarity.similarity_score
            ).filter(
                models.ItemSimilarity.movie_id == movie_id
            ).order_by(
                desc(models.ItemSimilarity.similarity_score)
            ).limit(count+10).all()
            
            collab_recommendations = [rec[0] for rec in collab_recs if rec[0] != movie_id]
        else:
            collab_recommendations = []
        
        # For hybrid, mix recommendations
        if method == 'hybrid':
            # Create a diverse list by alternating between the two methods
            mixed_recommendations = []
            for i in range(min(len(content_recommendations), len(collab_recommendations))):
                if i % 2 == 0 and content_recommendations:
                    mixed_recommendations.append(content_recommendations.pop(0))
                elif collab_recommendations:
                    mixed_recommendations.append(collab_recommendations.pop(0))
                    
            # Add any remaining recommendations
            mixed_recommendations.extend(content_recommendations[:count - len(mixed_recommendations)])
            mixed_recommendations.extend(collab_recommendations[:count - len(mixed_recommendations)])
            
            # Take just the requested count, removing duplicates
            recommendations = []
            for rec in mixed_recommendations:
                if rec not in recommendations and rec != movie_id:
                    recommendations.append(rec)
                if len(recommendations) >= count:
                    break
        elif method == 'content':
            recommendations = [rec for rec in content_recommendations if rec != movie_id][:count]
        else:  # collaborative
            recommendations = [rec for rec in collab_recommendations if rec != movie_id][:count]
        
        result = []
        for rec_id in recommendations:
            movie_info = get_movie_info(rec_id)
            
            # Generate appropriate explanation based on method
            if method == 'content':
                explanation = explain_content_recommendation(movie_id, rec_id)
            elif method == 'collaborative':
                explanation = explain_collaborative_recommendation(movie_id, rec_id)
            else:  # hybrid
                # Use content explanation for even indices, collaborative for odd
                if recommendations.index(rec_id) % 2 == 0:
                    explanation = explain_content_recommendation(movie_id, rec_id)
                else:
                    explanation = explain_collaborative_recommendation(movie_id, rec_id)
            
            movie_info['explanation'] = explanation
            result.append(movie_info)
        
        return jsonify({
            "baseMovie": get_movie_info(movie_id),
            "recommendations": result,
            "method": method
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/movies', methods=['GET'])
def get_movies():
    # Pagination parameters
    page = int(request.args.get('page', '1'))
    limit = int(request.args.get('limit', '20'))
    search = request.args.get('search', '').lower()
    
    db = get_db()
    
    # Filter movies
    query = db.query(models.Movie)
    if search:
        query = query.filter(func.lower(models.Movie.title).contains(search))
    
    # Get total count
    total = query.count()
    
    # Paginate results
    movies_list = query.offset((page - 1) * limit).limit(limit).all()
    
    # Format response
    movies = []
    for movie in movies_list:
        movies.append({
            'id': int(movie.movieId),
            'title': movie.title,
            'genres': movie.genres.split('|'),
            'year': movie.title[-5:-1] if movie.title[-5:-1].isdigit() else None,
        })
    
    return jsonify({
        "total": total,
        "page": page,
        "limit": limit,
        "movies": movies
    })

@app.route('/api/prepare', methods=['GET'])
def prepare_data():
    """Endpoint to rebuild similarity matrices - should be run after data is loaded"""
    
    try:
        db = get_db()
        
        # Get movies
        movies = db.query(models.Movie).all()
        
        # Create DataFrame from movies for easier processing
        movies_data = []
        for m in movies:
            movies_data.append({
                'movieId': m.movieId,
                'title': m.title,
                'genres': m.genres
            })
        
        movies_df = pd.DataFrame(movies_data)
        
        # Get ratings
        ratings = db.query(models.Rating).all()
        ratings_data = []
        for r in ratings:
            ratings_data.append({
                'userId': r.userId,
                'movieId': r.movieId,
                'rating': r.rating
            })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        # Create user-item matrix for collaborative filtering
        user_item_matrix = ratings_df.pivot(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
        # Generate collaborative filtering similarity matrix
        item_similarity = cosine_similarity(user_item_matrix.T)
        
        # Prepare movie content features
        movies_df['content_features'] = movies_df['title'] + ' ' + movies_df['genres'].str.replace('|', ' ')
        
        # Update content features in database
        for _, row in movies_df.iterrows():
            movie = db.query(models.Movie).filter(models.Movie.movieId == row['movieId']).first()
            if movie:
                movie.content_features = row['content_features']
        
        db.commit()
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['content_features'])
        content_similarity = cosine_similarity(tfidf_matrix)
        
        # Store similarity matrices in the database
        # First clear existing data
        db.query(models.ItemSimilarity).delete()
        db.query(models.ContentSimilarity).delete()
        db.commit()
        
        # Add collaborative similarity data
        for i, movie_id in enumerate(user_item_matrix.columns):
            for j, similar_id in enumerate(user_item_matrix.columns):
                if i != j and item_similarity[i, j] > 0.1:  # Only store significant similarities
                    db.add(models.ItemSimilarity(
                        movie_id=movie_id,
                        similar_movie_id=similar_id,
                        similarity_score=float(item_similarity[i, j])
                    ))
        
        # Add content similarity data
        for i, movie_id in enumerate(movies_df['movieId']):
            for j, similar_id in enumerate(movies_df['movieId']):
                if i != j and content_similarity[i, j] > 0.1:  # Only store significant similarities
                    db.add(models.ContentSimilarity(
                        movie_id=movie_id,
                        similar_movie_id=similar_id,
                        similarity_score=float(content_similarity[i, j])
                    ))
        
        db.commit()
        
        return jsonify({"message": "Data prepared successfully"})
    
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get basic stats about the dataset"""
    db = get_db()
    
    total_movies = db.query(func.count(models.Movie.id)).scalar()
    total_users = db.query(func.count(func.distinct(models.Rating.userId))).scalar()
    total_ratings = db.query(func.count(models.Rating.id)).scalar()
    avg_rating = db.query(func.avg(models.Rating.rating)).scalar()
    
    # Get unique genres
    movies = db.query(models.Movie.genres).all()
    all_genres = []
    for movie in movies:
        all_genres.extend(movie[0].split('|'))
    unique_genres = len(set(all_genres))
    
    stats = {
        "totalMovies": total_movies,
        "totalUsers": total_users, 
        "totalRatings": total_ratings,
        "avgRating": float(avg_rating) if avg_rating else 0,
        "uniqueGenres": unique_genres
    }
    
    return jsonify(stats)

@app.route('/api/setup', methods=['GET'])
def setup_database():
    """One-time endpoint to populate the database"""
    try:
        from app.migration import migrate_data
        migrate_data()
        return jsonify({"message": "Database setup complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))