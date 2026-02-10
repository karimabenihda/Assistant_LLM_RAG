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
from schemas import UserInDB, Qts, Token, UserLogin, QuestionInDB,QueryFullResponse

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
# ----- Load MLflow RAG model -----
import mlflow
import mlflow.pyfunc
from pathlib import Path

# 1. On définit le chemin absolu vers la base SQLite pour que MLflow la trouve
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DB_PATH = ROOT_DIR / "mlflow.db"

# Utilisation de l'URI absolue pour SQLite (obligatoire sur Windows)
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")

# 2. On définit le chemin direct vers le dossier du modèle (plus robuste que le Run ID)
# MODEL_PATH = ROOT_DIR / "models" / "rag_pyfunc" / "python_model.pkl"


# try:
#     if MODEL_PATH.exists():
#         print(f"Tentative de chargement local : {MODEL_PATH}")
#         rag_model = mlflow.pyfunc.load_model(str(MODEL_PATH))
#     else:
#         print("Dossier local non trouvé, tentative via Run ID...")
#         # run_id = "08afab7835534cfe906a85b0ce54875f"
#         # model_uri = f"runs:/{run_id}/rag_model"
#         # rag_model = mlflow.pyfunc.load_model(model_uri)
    
#     print("✅ Modèle RAG chargé avec succès !")
# except Exception as e:
#     print(f"❌ Erreur critique de chargement : {e}")
    
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
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# ----- Load MLflow RAG model -----
import mlflow.pyfunc
from pathlib import Path

MODEL_DIR = Path("../mlruns/1/08afab7835534cfe906a85b0ce54875f/artifacts/rag_model")

rag_model = None

try:
    # On vérifie si le dossier existe
    if MODEL_DIR.exists():
        print(f"Tentative de chargement local : {MODEL_DIR}")
        # On convertit en string pour mlflow
        rag_model = mlflow.pyfunc.load_model(str(MODEL_DIR))
        print(f"✅ Modèle chargé avec succès !")
    else:
        # Debug : affiche le chemin absolu pour voir où Python cherche vraiment
        print(f"❌ Erreur : Dossier introuvable à {MODEL_DIR.resolve()}")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")

@app.post("/query", response_model=QueryFullResponse)
def ask_rag(request_data: Qts, user=Depends(verify_token), db: Session = Depends(get_db)):
    if rag_model is None:
        raise HTTPException(status_code=503, detail="Modèle MLflow non chargé")

    # 1. Extraction des infos
    user_email = user.get("sub") # On récupère l'identifiant du token
    # On cherche l'ID numérique de l'utilisateur en DB
    db_user = db.query(User).filter(User.email == user_email).first()
    user_id = db_user.id if db_user else 0

    start_time = datetime.utcnow()

    # 2. Prédiction RAG
    question_text = request_data.qts
    question_df = pd.DataFrame([{"question": question_text}])
    
    try:
        prediction = rag_model.predict(question_df)
        # Gestion du format de sortie de predict (souvent une liste ou un array)
        answer_text = prediction[0] if isinstance(prediction, (list, pd.Series)) else prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

    end_time = datetime.utcnow()
    latency = int((end_time - start_time).total_seconds() * 1000)

    # 3. Création de l'objet pour la DB
    new_query = Query(
        userid=user_id,
        question=question_text,
        answer=str(answer_text),
        cluster=0, 
        latency_ms=latency,
        created_at=start_time
    )
    db.add(new_query)
    db.commit()
    db.refresh(new_query)

    # 4. Retour conforme à ta demande
    return {
        "qst_meta": QuestionInDB(
            userid=new_query.userid,
            question=new_query.question,
            answer=new_query.answer,
            cluster=new_query.cluster,
            latency_ms=new_query.latency_ms,
            created_at=new_query.created_at
        ),
        "extra_data": Qts(qts=new_query.question)
    }
    
# ----- User History Endpoint -----
@app.get("/history")
def get_user_history(user=Depends(verify_token), db: Session = Depends(get_db)):
    user_id = user.get("sub")
    return db.query(Query).filter(Query.userid == user_id).all()
