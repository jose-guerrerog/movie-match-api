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
    try:
        # Get parameters with defaults
        movie_id = request.args.get('movie_id', type=int, default=1)
        count = request.args.get('count', type=int, default=5)
        
        # Create database session
        from database import SessionLocal
        import models
        from sqlalchemy.sql.expression import func
        from sqlalchemy import desc
        
        db = SessionLocal()
        
        try:
            # Get the requested movie
            movie = db.query(models.Movie).filter(models.Movie.movieId == movie_id).first()
            
            if not movie:
                return jsonify({"error": "Movie not found"}), 404
            
            # Extract base movie information
            base_movie = {
                'id': movie.movieId,
                'title': movie.title,
                'genres': movie.genres.split('|') if movie.genres else [],
                'year': movie.title[-5:-1] if movie.title and len(movie.title) > 5 and movie.title[-5:-1].isdigit() else None
            }
            
            # Find users who rated this movie highly
            users_who_liked = db.query(models.Rating.userId).filter(
                models.Rating.movieId == movie_id,
                models.Rating.rating >= 4.0
            ).all()
            
            user_ids = [u[0] for u in users_who_liked]
            
            # Get recommendations based on collaborative filtering
            collab_recommendations = []
            
            if user_ids:
                # Find other movies these users rated highly
                movies_liked_by_same_users = db.query(
                    models.Rating.movieId,
                    func.count(models.Rating.userId).label('user_count'),
                    func.avg(models.Rating.rating).label('avg_rating')
                ).filter(
                    models.Rating.userId.in_(user_ids),
                    models.Rating.movieId != movie_id,
                    models.Rating.rating >= 4.0
                ).group_by(
                    models.Rating.movieId
                ).having(
                    func.count(models.Rating.userId) >= 2  # At least 2 users in common
                ).order_by(
                    desc('user_count'),
                    desc('avg_rating')
                ).limit(count).all()
                
                # Get movie details
                for movie_id, user_count, avg_rating in movies_liked_by_same_users:
                    collab_movie = db.query(models.Movie).filter(models.Movie.movieId == movie_id).first()
                    if collab_movie:
                        collab_recommendations.append({
                            'movie': collab_movie,
                            'user_count': user_count,
                            'avg_rating': avg_rating
                        })
            
            # Get genre-based recommendations as backup
            movie_genres = movie.genres.split('|') if movie.genres else []
            genre_recommendations = []
            
            for genre in movie_genres:
                # For each genre, find some matching movies
                genre_matches = db.query(models.Movie).filter(
                    models.Movie.movieId != movie_id,
                    models.Movie.genres.like(f'%{genre}%'),
                    ~models.Movie.movieId.in_([r['movie'].movieId for r in collab_recommendations])
                ).order_by(func.random()).limit(3).all()
                
                for match in genre_matches:
                    genre_recommendations.append({
                        'movie': match,
                        'genre': genre
                    })
            
            # Combine recommendations with priority for collaborative filtering
            final_recommendations = []
            
            # First add collaborative recommendations
            for rec in collab_recommendations:
                if len(final_recommendations) < count:
                    final_recommendations.append({
                        'movie': rec['movie'],
                        'explanation': f"{rec['user_count']} users who liked '{movie.title}' also rated this highly (avg rating: {rec['avg_rating']:.1f})"
                    })
            
            # Then add genre recommendations to fill remaining slots
            for rec in genre_recommendations:
                if len(final_recommendations) < count:
                    final_recommendations.append({
                        'movie': rec['movie'],
                        'explanation': f"This movie shares the genre '{rec['genre']}' with '{movie.title}'"
                    })
            
            # Format the output
            recommendations = []
            for rec in final_recommendations:
                other = rec['movie']
                other_genres = other.genres.split('|') if other.genres else []
                
                recommendations.append({
                    'id': other.movieId,
                    'title': other.title,
                    'genres': other_genres,
                    'year': other.title[-5:-1] if other.title and len(other.title) > 5 and other.title[-5:-1].isdigit() else None,
                    'explanation': rec['explanation']
                })
            
            # Return the results
            return jsonify({
                "baseMovie": base_movie,
                "recommendations": recommendations,
                "method": "hybrid"
            })
            
        except Exception as e:
            print(f"Inner error: {str(e)}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
        finally:
            db.close()
            
    except Exception as e:
        print(f"Outer error: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

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
        from migration import migrate_data
        migrate_data()
        return jsonify({"message": "Database setup complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simple-recommend', methods=['GET'])
def simple_recommend():
    movie_id = int(request.args.get('movie_id', '1'))
    count = int(request.args.get('count', '5'))
    method = request.args.get('method', 'content')  # 'content' or 'collaborative'
    
    db = SessionLocal()
    
    try:
        # Get the target movie
        movie = db.query(models.Movie).filter(models.Movie.movieId == movie_id).first()
        if not movie:
            return jsonify({"error": "Movie not found"}), 404
        
        recommendations = []
        
        if method == 'content':
            # Content-based: Find movies with similar genres
            movie_genres = set(movie.genres.split('|'))
            
            # Find movies with similar genres
            similar_movies = db.query(models.Movie).filter(
                models.Movie.movieId != movie_id
            ).all()
            
            # Score each movie by genre overlap
            movie_scores = []
            for similar in similar_movies:
                similar_genres = set(similar.genres.split('|'))
                overlap = len(movie_genres.intersection(similar_genres))
                if overlap > 0:  # Only include movies with at least one shared genre
                    movie_scores.append({
                        'movie': similar,
                        'score': overlap / len(movie_genres.union(similar_genres))  # Jaccard similarity
                    })
            
            # Sort by similarity score
            movie_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Get top recommendations
            for i, scored in enumerate(movie_scores[:count]):
                similar = scored['movie']
                recommendations.append({
                    'id': int(similar.movieId),
                    'title': similar.title,
                    'genres': similar.genres.split('|'),
                    'year': similar.title[-5:-1] if similar.title[-5:-1].isdigit() else None,
                    'explanation': f"This movie shares similar genres with '{movie.title}'"
                })
        
        elif method == 'collaborative':
            # Find users who rated this movie highly
            high_raters = db.query(models.Rating.userId).filter(
                models.Rating.movieId == movie_id,
                models.Rating.rating >= 4.0
            ).all()
            
            high_raters = [user[0] for user in high_raters]
            
            if not high_raters:
                # Fallback to content-based if no users rated this movie highly
                return simple_recommend()
            
            # Find other movies these users rated highly
            similar_movies = db.query(
                models.Rating.movieId,
                func.avg(models.Rating.rating).label('avg_rating'),
                func.count(models.Rating.userId).label('rating_count')
            ).filter(
                models.Rating.userId.in_(high_raters),
                models.Rating.movieId != movie_id,
                models.Rating.rating >= 4.0
            ).group_by(
                models.Rating.movieId
            ).having(
                func.count(models.Rating.userId) >= 2  # At least 2 users in common
            ).order_by(
                desc('rating_count'),
                desc('avg_rating')
            ).limit(count).all()
            
            # Get full movie details
            for i, (similar_id, avg_rating, rating_count) in enumerate(similar_movies):
                similar = db.query(models.Movie).filter(models.Movie.movieId == similar_id).first()
                if similar:
                    recommendations.append({
                        'id': int(similar.movieId),
                        'title': similar.title,
                        'genres': similar.genres.split('|'),
                        'year': similar.title[-5:-1] if similar.title[-5:-1].isdigit() else None,
                        'explanation': f"{rating_count} users who liked '{movie.title}' also rated this movie highly"
                    })
        
        return jsonify({
            "baseMovie": {
                'id': int(movie.movieId),
                'title': movie.title,
                'genres': movie.genres.split('|'),
                'year': movie.title[-5:-1] if movie.title[-5:-1].isdigit() else None,
            },
            "recommendations": recommendations,
            "method": method
        })
    
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))