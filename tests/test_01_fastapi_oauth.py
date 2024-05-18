from fastapi.testclient import TestClient
from pytest import CaptureFixture
from src.fastapi_oauth.fastapi_oauth import app

client = TestClient(app=app, base_url="http://localhost:8001")

def test_read_public_data():
    response = client.get("/")
    assert response.json() == {"message": "Hello World!"}

def test_unable_read_private_data():
    response = client.get("/secured")
    assert response.json() == {"detail": "Not authenticated"}
    
 
def test_get_token():
    data = {
        "username": "John",
        "password": "John",
    }

    response = client.post("/token", data=data)

    assert response.status_code == 200
    assert "access_token" in response.json()    
    
    
def test_login_wrong_password():
    response = client.post(
        "/token", 
        headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
            },
        data = {
            "grant_type": "password",
            "username": "Alice",
            "password": "John",
            "scope": "",
            "client_id": "string",
            "client_secret": "string"
            }
        )
    assert response.status_code == 400
    message = response.json()
    assert message["detail"] == "Incorrect username or password"
    
def test_login_wrong_username():
    response = client.post(
        "/token", 
        headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
            },
        data = {
            "grant_type": "password",
            "username": "Alicia",
            "password": "Alice",
            "scope": "",
            "client_id": "string",
            "client_secret": "string"
            }
        )
    assert response.status_code == 400
    message = response.json()
    assert message["detail"] == "Incorrect username or password"
    
    
def test_login_for_access_token(capsys: CaptureFixture[str]):
    response = client.post(
        "/token", 
        headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
            },
        data = {
            "grant_type": "password",
            "username": "Alice",
            "password": "Alice",
            "scope": "",
            "client_id": "string",
            "client_secret": "string"
            }
        )
    assert response.status_code == 200
    assert "access_token" in response.json()
    message = response.json()
    assert message["access_token"] == "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJBbGljZSJ9.NlaamftPNAgtReF0kY03XiDWplViB3DFfuqjnZ0Dy48"
    assert message["token_type"] == "bearer"
    
    
   


def test_read_private_data():
    response = client.get("/secured")
    assert response.json() == {"message": "Hello World, but secured!"}

    