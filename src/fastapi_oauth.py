import mysql.connector
from mysql.connector import Error
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from typing import Optional
import jwt
from datetime import datetime, timedelta

app = FastAPI()

# Configuration de la base de données MySQL
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = "Rakuten"
MYSQL_DB = "rakuten_db"

# Configuration pour le JWT
SECRET_KEY = "secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRATION = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

users_db = {

    "Olivier": {
        "username": "Olivier",
        "name": "Daniel Datascientest",
        "email": "daniel@datascientest.com",
        "password": pwd_context.hash('Olivier'),
        "resource" : "Module DE",
    },

    "johndatascientest" : {
        "username" :  "johndatascientest",
        "name" : "John Datascientest",
        "email" : "john@datascientest.com",
        "password" : pwd_context.hash('secret'),
        'resource' : 'Module DS',
    }
}

# Fonction pour interroger la base de données et récupérer les informations d'identification de l'utilisateur
def get_user(username: str):
    # return users_db.get(username)
    try:
        print("username : ",username)
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            port="3306", 
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        print("username : ",username)
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
        cursor.close()
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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), form_data: OAuth2PasswordRequestForm = Depends()):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.JWTError as e:
        raise credentials_exception

    user = get_user(username)
    if user is None:
        raise credentials_exception

    # Vérification du mot de passe
    # password = user.get("password")
    # if not verify_password(form_data.password, password):
    #     raise HTTPException(status_code=400, detail="Incorrect username or password")

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
    access_token = create_access_token(data={"sub": form_data.username}, expires_delta=access_token_expires)

    return {"access_token": access_token, "token_type": "bearer"}

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
def read_private_data(token: str = Depends(oauth2_scheme)):
# def read_private_data(current_user: str = Depends(get_current_user)):
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

    return {"message": "Hello World, but secured!"}

'''
@app.on_event("startup")
async def startup_event():
    global connection
    
    # Définir les options de connexion TCP
    client_flags = [mysql.connector.ClientFlag.PROTOCOL_41] #, mysql.connector.ClientFlag.SSL]
    # Connexion à la base de données au démarrage de l'application
    try:
        connection = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            port="3306", 
            database=MYSQL_DB,
            client_flags=client_flags, 
        )
        if connection.is_connected():
            print("Connected to MySQL Server")
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global connection
    
    # Fermeture de la connexion à la base de données à l'arrêt de l'application
    try:
        connection.close()
        print("Connection to MySQL Server closed")
    except Error as e:
        print(f"Error while closing connection to MySQL: {e}")
'''
@app.get("/prediction")
def prediction(token: str = Depends(oauth2_scheme)):
    global predictor
    
    # Appel au service d'authentification pour vérifier le token
    auth_response = requests.get("http://localhost:8000/secured", headers={"Authorization": f"Bearer {token}"})
    return {"message": auth_response.status_code}