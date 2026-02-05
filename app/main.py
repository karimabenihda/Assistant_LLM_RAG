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
import os
import mlflow

# 1. On récupère le chemin absolu proprement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# On remonte d'un dossier pour sortir de 'app' et trouver 'models'
ROOT_DIR = os.path.dirname(BASE_DIR)
local_path = os.path.join(ROOT_DIR, "models", "rag_pyfunc")

# 2. On transforme le chemin Windows en URI compatible MLflow (file:///C:/...)
# On remplace les \ par des / et on ajoute le préfixe
model_uri = f"file:///{local_path.replace(os.sep, '/')}"

print(f"Tentative de chargement via : {model_uri}")

try:
    rag_model = mlflow.pyfunc.load_model(model_uri)
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")
    rag_model = None
    
    
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
mlflow.set_tracking_uri("sqlite:///mlflow.db")
run_id = "08afab7835534cfe906a85b0ce54875f"
model_uri = f"runs:/{run_id}/rag_model"
    # model_uri = "models:/RAG_Pipeline/latest" 
    # rag_model = mlflow.pyfunc.load_model(model_uri)
    # rag_model = mlflow.pyfunc.load_model("models:/RAG_Pipeline/1")
try:
    rag_model = mlflow.pyfunc.load_model(model_uri)
    print("Succès : Modèle chargé via Run ID")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")
    rag_model = None

@app.post("/query")
def ask_rag(qst_meta: QuestionInDB, extra_data: Qts, user=Depends(verify_token), db: Session = Depends(get_db)):
    if rag_model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    user_id = user.get("sub")
    start_time = datetime.utcnow()

    # Préparation de l'input pour MLflow (DataFrame)
    question_df = pd.DataFrame([{"question": extra_data.qts}]) # Utilise le bon champ (qts)
    
    try:
        # MLflow predict retourne souvent un array ou une liste
        answer = rag_model.predict(question_df)
        if isinstance(answer, list) or hasattr(answer, "tolist"):
            answer = answer[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

    latency = (datetime.utcnow() - start_time).total_seconds() * 1000

    # Sauvegarde
    history = Query(
        userid=user_id,
        question=extra_data.qts,
        answer=str(answer),
        cluster=qst_meta.cluster,
        latency_ms=latency,
        created_at=datetime.utcnow()
    )
    db.add(history)
    db.commit()

    return {"result": answer, "user_id": user_id, "latency_ms": latency}

# ----- User History Endpoint -----
@app.get("/history")
def get_user_history(user=Depends(verify_token), db: Session = Depends(get_db)):
    user_id = user.get("sub")
    return db.query(Query).filter(Query.userid == user_id).all()
