import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

# Get the PostgreSQL connection string from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

# Render PostgreSQL connection fix
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with connection pool settings
engine = create_engine(
    DATABASE_URL,
    pool_size=5,  # Start with 5 connections
    max_overflow=10,  # Allow up to 10 more connections
    pool_timeout=30,  # Timeout waiting for a connection (seconds)
    pool_recycle=1800,  # Recycle connections after 30 minutes
    pool_pre_ping=True  # Test connections with a ping before using
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency to get the database session"""
    db = SessionLocal()
    try:
        return db
    except:
        db.close()
        raise