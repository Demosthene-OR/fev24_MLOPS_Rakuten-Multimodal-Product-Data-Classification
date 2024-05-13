from unittest.mock import patch, MagicMock
from src.main import main

@patch('src.main.DataImporter')
@patch('src.main.TextPreprocessor')
@patch('src.main.ImagePreprocessor')
@patch('src.main.TextRnnModel')
@patch('src.main.ImageVGG16Model')
@patch('src.main.load_model')
def test_main(mock_load_model, mock_ImageVGG16Model, mock_TextRnnModel,
              mock_ImagePreprocessor, mock_TextPreprocessor, mock_DataImporter):
    mock_args = MagicMock()
    mock_args.x_train_path = "dummy_x_train_path.csv"
    mock_args.y_train_path = "dummy_y_train_path.csv"
    mock_args.images_path = "dummy_images_path"
    mock_args.model_path = "dummy_model_path"
    mock_args.n_epochs = 1
    mock_args.samples_per_class = 50
    mock_args.with_test = 0
    mock_args.random_state = 42

    mock_instance = MagicMock()
    mock_DataImporter.return_value = mock_instance
    mock_instance.load_data.return_value = dummy_dataframe
    mock_instance.split_train_test.return_value = (dummy_X_train, dummy_X_val, dummy_X_test, dummy_y_train, dummy_y_val, dummy_y_test)

    main()

    mock_DataImporter.assert_called_once_with("dummy_x_train_path.csv", "dummy_y_train_path.csv", "dummy_model_path")
    mock_instance.load_data.assert_called_once()
    mock_instance.split_train_test.assert_called_once_with(dummy_dataframe, samples_per_class=50, random_state=42, with_test=False)
    mock_TextPreprocessor.return_value.preprocess_text_in_df.assert_called()
    mock_ImagePreprocessor.return_value.preprocess_images_in_df.assert_called()
    mock_TextRnnModel.assert_called_once_with(file_path="dummy_model_path")
    mock_TextRnnModel.return_value.preprocess_and_fit.assert_called()
    mock_ImageVGG16Model.assert_called_once_with(file_path="dummy_model_path")
    mock_ImageVGG16Model.return_value.preprocess_and_fit.assert_called()
    mock_load_model.assert_called()
