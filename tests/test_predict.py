import unittest
from unittest.mock import patch, MagicMock

# import sys

# sys.path.append("C:\Users\hp\Documents\GitHub\fev24_MLOPS_Rakuten-Multimodal-Product-Data-Classification\src")

from src.predict import main

@patch('predict.Predict')
@patch('predict.load_model')
def test_main(mock_load_model, mock_Predict):
    mock_args = MagicMock()
    mock_args.dataset_path = "dummy_dataset_path.csv"
    mock_args.images_path = "dummy_images_path"
    mock_args.prediction_path = "dummy_prediction_path.csv"

    mock_instance = MagicMock()
    mock_Predict.return_value = mock_instance

    main()

    mock_Predict.assert_called_once_with(
        tokenizer=mock_load_model.return_value,
        rnn=mock_load_model.return_value,
        vgg16=mock_load_model.return_value,
        best_weights=mock_load_model.return_value,
        mapper=mock_load_model.return_value,
        filepath=mock_args.dataset_path,
        imagepath=mock_args.images_path
    )

    mock_instance.predict.assert_called_once()
    mock_instance.predict.return_value.to_csv.assert_called_once_with(mock_args.prediction_path, index=False)
