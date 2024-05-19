from fastapi.testclient import TestClient
from src.main_API import app

client = TestClient(app=app, base_url="http://localhost:8002")

def test_main_unauthorized():
    ACCESS_TOKEN_AUTH_1= "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJKb2huIn0.HW2PSY6qVAPqtOi49Kf-bHh52e30BmvdQmqiC25KctY"
    input_data = {
        "x_train_path": "data/preprocessed/X_train_update.csv",
        "y_train_path": "data/preprocessed/Y_train_CVw08PX.csv",
        "images_path": "data/preprocessed/image_train",
        "model_path": "models",
        "n_epochs": 1,
        "samples_per_class": 50,
        "with_test": False,
        "random_state": 42,
        "api_secured": False
        }
    response = client.post("/train", headers={"Authorization": f"Bearer {ACCESS_TOKEN_AUTH_1}"}, data=input_data)
    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
    

def test_main():
    ACCESS_TOKEN_AUTH_2= "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJGYWRpbWF0b3UifQ.r43zrSm_B3l5-xNjf7Q9XZXOQncGuI9YzarapOA0Wgg"
    input_data = {
        "x_train_path": "data/preprocessed/X_train_update.csv",
        "y_train_path": "data/preprocessed/Y_train_CVw08PX.csv",
        "images_path": "data/preprocessed/image_train",
        "model_path": "models",
        "n_epochs": 1,
        "samples_per_class": 50,
        "with_test": False,
        "random_state": 42,
        "api_secured": False
        }
    response = client.post("/train", headers={"Authorization": f"Bearer {ACCESS_TOKEN_AUTH_2}"}, data=input_data)
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
        
