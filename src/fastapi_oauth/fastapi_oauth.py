import mysql.connector
from mysql.connector import Error
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from typing import Optional
import jwt
from jwt import PyJWTError
from datetime import datetime, timedelta, timezone

app = FastAPI()

class UserDetail(BaseModel):
    FirstName: str
    LastName: str
    Email: str
    Authorization: int # integer, not str
    username: str
    password: str # The UserDetail model includes a password field


# Configuration de la base de données MySQL
MYSQL_HOST = "users_db"
MYSQL_USER = "root"
MYSQL_PASSWORD = "Rakuten"
MYSQL_DB = "rakuten_db"

# Configuration pour le JWT
SECRET_KEY = "secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRATION = 45

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fonction pour interroger la base de données et récupérer les informations d'identification de l'utilisateur
def get_user(username: str):
    connection = None
    cursor = None
    # return users_db.get(username)
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port="3306", 
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            query = f"SELECT * FROM Users WHERE username = '{username}'"
            cursor.execute(query)
            user_connected = cursor.fetchone()
            print("userconnected : ",user_connected)
            return user_connected
    except Error as e:
        print(f"Error while querying MySQL: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


def create_user(user: UserDetail):
    hashed_password = pwd_context.hash(user.password)
    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port="3306",
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        if connection.is_connected():
            cursor = connection.cursor()
            query = "INSERT INTO Users (FirstName, LastName, Email, Authorization, username, password) VALUES (%s, %s, %s, %s, %s, %s)"
            cursor.execute(query, (user.FirstName, user.LastName, user.Email, user.Authorization, user.username, hashed_password))
            connection.commit()
    except Error as e:
        print(f"Error while querying MySQL: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()
            
def delete_user(username: str):
    connection = None
    cursor = None
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port="3306",
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        if connection.is_connected():
            cursor = connection.cursor()
            query = "DELETE FROM Users WHERE username = %s"
            cursor.execute(query, (username,))
            connection.commit()
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="User not found")
    except Error as e:
        print(f"Error while querying MySQL: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
        to_encode.update({"exp": expire})
    else:
        expire = None # datetime.now(timezone.utc) + timedelta(minutes=15)
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(jwt=token, key=SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except PyJWTError as e:
        raise credentials_exception
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Description:
    Cette route permet à un utilisateur de s'authentifier en fournissant un nom d'utilisateur et un mot de passe. Si l'authentification est réussie, elle renvoie un jeton d'accès JWT.

    Args:
    - form_data (OAuth2PasswordRequestForm, dépendance): Les données de formulaire contenant le nom d'utilisateur et le mot de passe.

    Returns:
    - Token: Un modèle de jeton d'accès JWT.

    Raises:
    - HTTPException(400, detail="Incorrect username or password"): Si l'authentification échoue en raison d'un nom d'utilisateur ou d'un mot de passe incorrect, une exception HTTP 400 Bad Request est levée.
    """
    user = get_user(form_data.username) 
    hashed_password = user["password"]
    if not user or not verify_password(form_data.password, hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRATION)
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=None) #, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}

# A new endpoint POST /users is created to handle user registration.
# The create_new_user function receives a UserDetail object, 
# checks if the username already exists, hashes the password, 
# and inserts the new user into the MySQL database using the create_user function.
@app.post("/usercreate", response_model=UserDetail)
async def create_new_user(user: UserDetail):
    existing_user = get_user(user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    create_user(user)
    return user

@app.post("/userdelete/{username}", status_code=204)
async def delete_existing_user(username: str):
    existing_user = get_user(username)
    if not existing_user:
        raise HTTPException(status_code=404, detail="User not found")
    delete_user(username)
    return {"detail": "User deleted successfully"}

@app.get("/")
def read_public_data():
    """
    Description:
    Cette route renvoie un message "Hello World!".

    Args:
    Aucun argument requis.

    Returns:
    - JSON: Renvoie un JSON contenant un message de salutation.

    Raises:
    Aucune exception n'est levée.
    """

    return {"message": "Hello World!"}

@app.get("/secured")
def read_private_data(current_user: dict = Depends(get_current_user)):
    """
    Description:
    Cette route renvoie un message "Hello World, but secured!" uniquement si l'utilisateur est authentifié.

    Args:
    - current_user (str, dépendance): Le nom d'utilisateur de l'utilisateur actuellement authentifié.

    Returns:
    - JSON: Renvoie un JSON contenant un message de salutation sécurisé si l'utilisateur est authentifié, sinon une réponse non autorisée.

    Raises:
    - HTTPException(401, detail="Unauthorized"): Si l'utilisateur n'est pas authentifié, une exception HTTP 401 Unauthorized est levée.
    """

    return {**current_user}

'''
@app.on_event("startup")
async def startup_event():

@app.on_event("shutdown")
async def shutdown_event():

'''
