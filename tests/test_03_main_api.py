from fastapi.testclient import TestClient
from src.main_API import app

client = TestClient(app=app, base_url="http://localhost:8002")

def test_main():
    headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
            },
    # TODO
    input_data = {
            "grant_type": "password",
            "username": "Alice",
            "password": "Alice",
            "scope": "",
            "client_id": "string",
            "client_secret": "string"
            }
    response = client.post("/train", headers=headers, data=input_data)
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
    
    
