import os
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def download_movielens():
    """Download the MovieLens small dataset (100k)"""
    print("Downloading MovieLens dataset...")
    
    # MovieLens 100K dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the zip file
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(DATA_DIR)
        
        # Move files from the extracted directory to our data directory
        extracted_dir = os.path.join(DATA_DIR, "ml-latest-small")
        for filename in os.listdir(extracted_dir):
            if filename.endswith(".csv"):
                src = os.path.join(extracted_dir, filename)
                dst = os.path.join(DATA_DIR, filename)
                os.rename(src, dst)
                
        # Remove the extracted directory
        for root, dirs, files in os.walk(extracted_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(extracted_dir)
        
        print("Dataset downloaded and extracted successfully!")
        return True
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def prepare_data():
    """Prepare the dataset for the recommendation system"""
    print("Preparing dataset...")
    
    # Check if the required files exist
    if not os.path.exists(os.path.join(DATA_DIR, 'movies.csv')) or \
       not os.path.exists(os.path.join(DATA_DIR, 'ratings.csv')):
        print("Required files not found. Downloading...")
        if not download_movielens():
            return False
    
    try:
        # Load data
        movies_df = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
        ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
        
        print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        
        # Create user-item matrix for collaborative filtering
        print("Creating user-item matrix...")
        user_item_matrix = ratings_df.pivot(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
        # Create item-item similarity matrix
        print("Computing item-item similarity matrix...")
        item_similarity = cosine_similarity(user_item_matrix.T)
        print(f"Item similarity matrix shape: {item_similarity.shape}")
        
        # Prepare movie content features for content-based filtering
        print("Preparing content features...")
        movies_df['content_features'] = movies_df['title'] + ' ' + movies_df['genres'].str.replace('|', ' ')
        
        # Create TF-IDF matrix
        print("Computing TF-IDF features...")
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_df['content_features'])
        
        # Create content-based similarity matrix
        print("Computing content-based similarity matrix...")
        content_similarity = cosine_similarity(tfidf_matrix)
        print(f"Content similarity matrix shape: {content_similarity.shape}")
        
        # Save matrices for faster loading
        print("Saving matrices...")
        with open(os.path.join(DATA_DIR, 'user_item_matrix.pkl'), 'wb') as f:
            pickle.dump(user_item_matrix, f)
        with open(os.path.join(DATA_DIR, 'item_similarity.pkl'), 'wb') as f:
            pickle.dump(item_similarity, f)
        with open(os.path.join(DATA_DIR, 'content_similarity.pkl'), 'wb') as f:
            pickle.dump(content_similarity, f)
            
        print("Data preparation complete!")
        return True
    
    except Exception as e:
        print(f"Error preparing data: {e}")
        return False

if __name__ == "__main__":
    prepare_data()