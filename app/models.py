from sqlalchemy import Column, Integer, String, Float, Table, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

class Movie(Base):
    __tablename__ = "movies"
    
    id = Column(Integer, primary_key=True, index=True)
    movieId = Column(Integer, unique=True, index=True)
    title = Column(String)
    genres = Column(String)
    content_features = Column(String, nullable=True)
    
class Rating(Base):
    __tablename__ = "ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    userId = Column(Integer, index=True)
    movieId = Column(Integer, index=True)
    rating = Column(Float)
    timestamp = Column(Integer, nullable=True)

# Tables for storing similarity matrices
class ItemSimilarity(Base):
    __tablename__ = "item_similarities"
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, index=True)
    similar_movie_id = Column(Integer, index=True)
    similarity_score = Column(Float)

class ContentSimilarity(Base):
    __tablename__ = "content_similarities"
    
    id = Column(Integer, primary_key=True, index=True)
    movie_id = Column(Integer, index=True)
    similar_movie_id = Column(Integer, index=True)
    similarity_score = Column(Float)