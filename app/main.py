from schemas import UserInDB, Qts, Token, UserLogin, QuestionInDB
from models import User, Query
from passlib.context import CryptContext
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import jwt, JWTError
import os
from database import get_db, engine
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import numpy as np
from sklearn.cluster import KMeans
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import mlflow.pyfunc

# Load env
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
GEMINI_API_KEY = os.getenv("Gemini_API_Key")
os.environ["Gemini_API_Key"] = GEMINI_API_KEY

app = FastAPI()
security = HTTPBearer()

# Create tables
from models import Base
Base.metadata.create_all(bind=engine)

# ----- Helpers -----
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: int = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta or ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ----- RAG setup -----
# Load documents
loader = PyPDFLoader("../data/data.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in chunks]
embedding_vectors = embeddings.embed_documents(texts)
embedding_vectors = np.array(embedding_vectors)

# Optional clustering
kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
clusters = kmeans.fit_predict(embedding_vectors)

# Persist vectorstore
persist_dir = "../data/chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Prompt
prompt_template = """
You are a factual assistant.

You MUST answer the question using ONLY the information explicitly present in the context.

Context:
{context}

Question:
{question}

Rules:
- If the answer is NOT fully contained in the context, reply exactly with: "I don't know."
- Do NOT use prior knowledge.
- Do NOT guess or infer.
- Do NOT add explanations, assumptions, or extra details.
- If the context is empty or irrelevant, say: "I don't know."

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# ----- MLflow PyFunc wrapper -----
class RAGModel(mlflow.pyfunc.PythonModel):
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def predict(self, model_input, params=None):
        """
        model_input: pandas DataFrame or dict with column/key 'question'
        """
        if isinstance(model_input, dict):
            question = model_input.get("question")
        else:  # assume DataFrame
            question = model_input.iloc[0]["question"]

        if not question:
            return "I don't know."
        result = self.qa_chain({"query": question})
        return result.get("result", "I don't know.")

# Save RAG PyFunc
model_path = "../models/rag_pyfunc"
# mlflow.pyfunc.save_model(path=model_path, python_model=RAGModel(qa_chain))
# print(f"RAG pipeline saved to {model_path}")

# ----- Auth -----
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

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

# ----- RAG Query endpoint using saved PyFunc -----
import pandas as pd
loaded_rag = mlflow.pyfunc.load_model(model_path)

@app.post("/query")
def ask_rag(qst_meta: QuestionInDB, extra_data: Qts, user=Depends(verify_token), db: Session = Depends(get_db)):
    user_id = user.get("sub")
    start_time = datetime.utcnow()
    # Use the saved PyFunc model
    question_df = pd.DataFrame([{"question": extra_data.qst}])
    answer = loaded_rag.predict(question_df)
    end_time = datetime.utcnow()
    latency = (end_time - start_time).total_seconds() * 1000

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

    return {"result": answer, "user_id": user_id}

@app.get("/history")
def get_user_history(user=Depends(verify_token), db: Session = Depends(get_db)):
    user_id = user.get("sub")
    return db.query(Query).filter(Query.userid == user_id).all()