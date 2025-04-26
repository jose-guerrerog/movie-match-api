# migration.py
import pandas as pd
import os
from database import SessionLocal, engine, Base
import models

def migrate_data():
    """Import MovieLens data into PostgreSQL database"""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Check if data is already loaded
        if db.query(models.Movie).count() > 0:
            print("Data already exists in database!")
            return
        
        # Load data from CSV files
        data_dir = 'data'
        movies_df = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
        ratings_df = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
        
        print(f"Importing {len(movies_df)} movies...")
        
        # Add movies
        for _, row in movies_df.iterrows():
            movie = models.Movie(
                movieId=int(row['movieId']),
                title=row['title'],
                genres=row['genres'],
                content_features=f"{row['title']} {row['genres'].replace('|', ' ')}"
            )
            db.add(movie)
        
        db.commit()
        print("Movies imported successfully!")
        
        # Add ratings in batches
        print(f"Importing {len(ratings_df)} ratings...")
        batch_size = 10000
        for i in range(0, len(ratings_df), batch_size):
            batch = ratings_df.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
                rating = models.Rating(
                    userId=int(row['userId']),
                    movieId=int(row['movieId']),
                    rating=float(row['rating']),
                    timestamp=int(row['timestamp']) if 'timestamp' in row else None
                )
                db.add(rating)
            
            db.commit()
            print(f"Imported {i+len(batch)} ratings...")
        
        print("Ratings imported successfully!")
        
        # Now prepare the similarity matrices
        print("Building recommendation matrices...")
        from app import prepare_data
        prepare_data()
        
        print("Data migration complete!")
        
    except Exception as e:
        db.rollback()
        print(f"Error during migration: {str(e)}")
    finally:
        db.close()

if __name__ == '__main__':
    migrate_data()