import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
SECRET_KEY = os.getenv('SECRET_KEY', 'dev_secret_key')

# Data paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MOVIELENS_DIR = os.path.join(DATA_DIR, 'ml-latest-small')

# Model settings
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CONTENT_SIMILARITY_MATRIX_PATH = os.path.join(MODEL_DIR, 'content_similarity.npz')
COLLABORATIVE_MODEL_PATH = os.path.join(MODEL_DIR, 'collaborative_model.joblib')

# API settings
API_PREFIX = '/api/v1'
DEFAULT_RECOMMENDATIONS_COUNT = 10
MAX_RECOMMENDATIONS_COUNT = 50

# Recommendation system settings
CONTENT_FEATURES = ['genres']
CONTENT_WEIGHT = 0.4  # Weight for content-based recommendations in hybrid approach
COLLABORATIVE_WEIGHT = 0.6  # Weight for collaborative recommendations in hybrid approach

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)