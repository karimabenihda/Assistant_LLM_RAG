from schemas import UserInDB, Qts,Token, UserLogin,QuestionInDB
from models import User,Query
from passlib.context import CryptContext
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import jwt,JWTError
import os
from database import get_db
from dotenv import load_dotenv 
from models import Base
from database import engine
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma 
# from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import numpy as np
from sklearn.cluster import KMeans
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate


load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))  # convert to int

Gemini_API_Key=os.getenv('Gemini_API_Key')

app = FastAPI()
security = HTTPBearer()



Base.metadata.create_all(bind=engine)


# ----- Helpers -----
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
def hash_password(password: str) -> str:
    hashed = pwd_context.hash(password)
    return hashed


def verify_password(plain_password: str, hashed_password: str) -> bool:
    result = pwd_context.verify(plain_password, hashed_password)
    return result


def create_access_token(data: dict, expires_delta: int = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta or ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


#-----rag helper fcts-------
loader = PyPDFLoader("../data/data.pdf")
documents = loader.load()

text_splitter=RecursiveCharacterTextSplitter(
   chunk_size=400,  # Taille max d’un morceau
    chunk_overlap=100   #Texte répété entre chunks pour le contexte
)
chunks = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts = [chunk.page_content for chunk in chunks]
embedding_vectors = embeddings.embed_documents(texts)
embedding_vectors = np.array(embedding_vectors)

kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto")
clusters = kmeans.fit_predict(embedding_vectors)

persist_dir = "../data/chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)


os.environ["Gemini_API_Key"] = Gemini_API_Key

# Create Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

from langchain_core.prompts import PromptTemplate

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

    
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
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



# ----- Login -----
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

def verify_token(credentials:HTTPAuthorizationCredentials=Depends(security)  ):

    try:
        payload=jwt.decode(credentials.credentials,SECRET_KEY,  algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/query")
def ask_rag(qst_meta: QuestionInDB, extra_data: Qts,user=Depends(verify_token),db: Session = Depends(get_db)):
    user_id = user.get("sub")
    start_time = datetime.utcnow()
    result = qa_chain({"query":  extra_data.qst})
    answer=result.get('result')
    end_time = datetime.utcnow()
    latency = (end_time - start_time).total_seconds() * 1000
    history=Query(
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
    
@app.get('/history')
def get_user_history(user=Depends(verify_token), db: Session = Depends(get_db)):
    user_id = user.get("sub")    
    return db.query(Query).filter(Query.userid == user_id).all()