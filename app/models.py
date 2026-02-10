from sqlalchemy import Column,ForeignKey ,String, Integer, DateTime
from datetime import datetime
from app.database import Base
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    firstname = Column(String(50), nullable=False)
    lastname = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    queries= relationship("Query", back_populates="owner")
    
    
class Query(Base):
    __tablename__="queries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    userid = Column(Integer, ForeignKey("users.id"))
    question = Column(String(50), nullable=False)
    answer = Column(String(50), nullable=False)
    cluster = Column(Integer, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
        
    owner=relationship("User",back_populates="queries")