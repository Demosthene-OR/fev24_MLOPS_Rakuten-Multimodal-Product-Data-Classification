from fastapi.testclient import TestClient
from src.predict_API import app

client = TestClient(app=app, base_url="http://localhost:8000")

def test_initialisation():
    response = client.get("/initialisation")
    assert response.status_code == 200
    message = response.json()
    print(message)
    assert message["message"] == 'Initialisation effectuée avec succès'
    
def test_prediction_unSecured():
    input_prediction = {
        "dataset_path": "data/predict/X_test_update.csv",
        "images_path": "data/predict/image_test",
        "prediction_path": "data/predict",
        "api_secured": False
    }
    response = client.post("/prediction", data=input_prediction)
    assert response.status_code == 200
    
def test_prediction_Secured():
    input_data = {
        "dataset_path": "data/predict/X_test_update.csv",
        "images_path": "data/predict/image_test",
        "prediction_path": "data/predict",
        "api_secured": True
    }
    # TODO
    response = client.post("/prediction", data=input_data)
    assert response.status_code == 200
    