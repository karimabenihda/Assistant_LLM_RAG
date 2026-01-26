from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserInDB(BaseModel):
    firstname: str
    lastname: str
    email: EmailStr
    password: str
    created_at:datetime
    
class UserLogin(BaseModel):
    email: str
    password: str
    
    
class Token(BaseModel):
    access_token: str
    token_type: str