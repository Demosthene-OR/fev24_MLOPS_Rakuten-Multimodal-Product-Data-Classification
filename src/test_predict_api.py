# test_predict.py

import pytest
from predict_API import Predict  # Replace with the actual module name

# Mock data for testing
class MockPredictor:
    def predict(self, *args, **kwargs):
        # Return mock predictions
        return [0, 1, 2]

@pytest.fixture
def mock_predictor():
    return MockPredictor()

@pytest.fixture
def predict_instance(mock_predictor):
    # Initialize Predict with mock dependencies
    return Predict(
        tokenizer=None,
        rnn=None,
        vgg16=None,
        best_weights=None,
        mapper=None,
        filepath=None,
        imagepath=None,
        predictor=mock_predictor
    )

def test_predict_method(predict_instance):
    # Mock input data 
    input_data = {
        "description": ["véhicule vintage de la saga star wars le A-AST5", "Original Barbie Doll. Contains glitter lotion.."],
        "image_path": ["image_978593209_product_279822475.jpg", "image_1199384348_product_3228900895.jpg"] # replace with path to image
    }

    # Call the predict method
    results_df = predict_instance.predict(input_data)

    # Assertions
    assert len(results_df) == 2  # Check the number of rows
    assert "cat_pred" in results_df.columns  # Check column names
    assert "Category1" in results_df.columns  # Check other columns

    # You can add more specific assertions based on your actual data

def test_initialization_endpoint(client):
    # Assuming you have a FastAPI app instance (replace with actual app)
    response = client.get("/initialisation")

    assert response.status_code == 200
    assert response.json() == {"message": "Initialisation effectuée avec succès"}
