from pydantic import BaseModel
from datetime import datetime

class UserInDB(BaseModel):
    firstname: str
    lastname: str
    email: str
    password: str
    created_at:datetime
    
class UserLogin(BaseModel):
    email: str
    password: str
    
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class Qts(BaseModel):
    qts: str

class QuestionInDB(BaseModel):
    userid: int
    question: str
    answer: str
    cluster: int
    latency_ms: int
    created_at: datetime

# Ce modèle regroupe les deux pour la réponse finale
class QueryFullResponse(BaseModel):
    qst_meta: QuestionInDB
    extra_data: Qts