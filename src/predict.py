from features.build_features import TextPreprocessor
from features.build_features import ImagePreprocessor
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras
import pandas as pd
import argparse
from keras import backend as K
from tools import f1_m, load_model
import time

MAX_ROW = 15000
   
class Predict:
    def __init__(
        self,
        tokenizer,
        rnn,
        vgg16,
        best_weights,
        mapper,
        filepath,
        imagepath
    ):
        self.tokenizer = tokenizer
        self.rnn = rnn
        self.vgg16 = vgg16
        self.best_weights = best_weights
        self.mapper = mapper
        self.filepath = filepath
        self.imagepath = imagepath

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self):
        X = pd.read_csv(self.filepath)[:MAX_ROW] 
        # Remplacer les NaN par des chaînes vides
        X["designation"] = X["designation"].fillna('')
        X["description"] = X["description"].fillna('')
        X["description"] = X["designation"] + " " + X["description"]
        
        text_preprocessor = TextPreprocessor()
        image_preprocessor = ImagePreprocessor(self.imagepath)
        text_preprocessor.preprocess_text_in_df(X, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X)

        sequences = self.tokenizer.texts_to_sequences(X["description"])
        padded_sequences = pad_sequences(
            sequences, maxlen=50, padding="post", truncating="post"
        )

        target_size = (224, 224, 3)
        images = X["image_path"].apply(lambda x: self.preprocess_image(x, target_size))
        images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        rnn_proba = self.rnn.predict([padded_sequences])
        vgg16_proba = self.vgg16.predict([images])

        concatenate_proba = (
            self.best_weights[0] * rnn_proba + self.best_weights[1] * vgg16_proba
        )
        # print(concatenate_proba)
        final_predictions = np.argmax(concatenate_proba, axis=1)

        # Récupérer les noms des catégories à partir du mapper
        categories = list(self.mapper.values())

        # Créer un DataFrame pour stocker les résultats de la prédiction
        results_df = pd.DataFrame(columns=['cat_pred'] + categories)

        for i in range(len(final_predictions)):
            # Récupérer la catégorie prédite
            cat_pred = self.mapper[str(final_predictions[i])]

            # Récupérer les probabilités pour chaque catégorie
            proba_values = concatenate_proba[i]

            # Créer une ligne pour cette prédiction
            prediction_row = [cat_pred] + list(proba_values)

            # Ajouter la ligne au DataFrame
            results_df.loc[i] = prediction_row

        return results_df


def main():
    parser = argparse.ArgumentParser(description= "Input data")
    
    parser.add_argument("--dataset_path", default = "data/predict/X_test_update.csv", type=str,help="File path for the input CSV file.")
    parser.add_argument("--images_path", default = "data/predict/image_test", type=str,  help="Base path for the images.")
    parser.add_argument("--prediction_path", default = "data/predict/predictions.csv",  type=str,  help="Path for the prediction results.")
    args = parser.parse_args()

    # Charger les configurations et modèles
    with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    rnn = load_model("models","best_rnn_model.h5")
    vgg16 = load_model("models","best_vgg16_model.h5")

    with open("models/best_weights.json", "r") as json_file:
        best_weights = json.load(json_file)

    with open("models/mapper.json", "r") as json_file:
        mapper = json.load(json_file)
        
    
    predictor = Predict(
        tokenizer=tokenizer,
        rnn=rnn,
        vgg16=vgg16,
        best_weights=best_weights,
        mapper=mapper,
        filepath= args.dataset_path,
        imagepath = args.images_path,
    )

    # Création de l'instance Predict et exécution de la prédiction
    t_debut = time.time()
    predictions = predictor.predict()
    
    # Sauvegarde des prédictions
    # with open("data/preprocessed/predictions.json", "w", encoding="utf-8") as json_file:
    #     json.dump(predictions, json_file, indent=2)
    predictions.to_csv(args.prediction_path, index=False)
    
    t_fin = time.time()
    print("Durée de la prédiction : {:.2f}".format(t_fin - t_debut))

if __name__ == "__main__":
    main()