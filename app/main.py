# app/main.py
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
import pandas as pd
import mlflow.pyfunc

from database import get_db, engine
from models import Base, User, Query
from schemas import UserInDB, Qts, Token, UserLogin, QuestionInDB

# ----- Load environment -----
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
GEMINI_API_KEY = os.getenv("Gemini_API_Key")
os.environ["Gemini_API_Key"] = GEMINI_API_KEY

# ----- FastAPI setup -----
app = FastAPI()
security = HTTPBearer()

# ----- Create tables -----
Base.metadata.create_all(bind=engine)

# ----- Password helpers -----
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ----- JWT helpers -----
def create_access_token(data: dict, expires_delta: int = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta or ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ----- Load MLflow RAG model from Registry -----
# Make sure you have logged and registered your model as "RAG_Pipeline"
rag_model = mlflow.pyfunc.load_model("models:/RAG_Pipeline/Production")

# ----- Auth Endpoints -----
@app.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/register")
def register(user: UserInDB, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    new_user = User(
        firstname=user.firstname,
        lastname=user.lastname,
        email=user.email,
        password=hash_password(user.password),
        created_at=datetime.utcnow(),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user_id": new_user.id}

# ----- RAG Query Endpoint -----
@app.post("/query")
def ask_rag(qst_meta: QuestionInDB, extra_data: Qts, user=Depends(verify_token), db: Session = Depends(get_db)):
    user_id = user.get("sub")
    start_time = datetime.utcnow()

    # Convert question to DataFrame
    question_df = pd.DataFrame([{"question": extra_data.qst}])
    
    # Get answer from RAG PyFunc model
    answer = rag_model.predict(question_df)

    end_time = datetime.utcnow()
    latency = (end_time - start_time).total_seconds() * 1000  # in ms

    # Save query history
    history = Query(
        userid=user_id,
        question=extra_data.qts,
        answer=answer,
        cluster=qst_meta.cluster,
        latency_ms=latency,
        created_at=datetime.utcnow()
    )
    db.add(history)
    db.commit()
    db.refresh(history)

    return {"result": answer, "user_id": user_id, "latency_ms": latency}

# ----- User History Endpoint -----
@app.get("/history")
def get_user_history(user=Depends(verify_token), db: Session = Depends(get_db)):
    user_id = user.get("sub")
    return db.query(Query).filter(Query.userid == user_id).all()
