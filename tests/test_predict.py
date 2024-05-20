from fastapi.testclient import TestClient
from src.predict_API import app

client = TestClient(app=...)

def test_initialisation_endpoint():
    response = client.get("/initialisation")
    assert response.status_code == 200
    assert response.json() == {"message": "Initialisation effectuée avec succès"}

""" def test_prediction_endpoint():
    input_data = {
        "dataset_path": "data/predict/X_test_update.csv",
        "images_path": "data/predict/image_test",
        "prediction_path": "data/predict",
        "api_secured": False
    }
    response = client.post("/prediction", json=input_data)
    assert response.status_code == 200
    assert response.message == "Prédiction effectuée avec succès, demandée par un utilisateur inconnu"
    #"duration": 2.0526010990142822#
    # Add more assertions based on the expected response """
