from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import requests
from src.features.build_features import TextPreprocessor
from src.features.build_features import ImagePreprocessor
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import json
from tensorflow import keras
from keras import backend as K
from src.tools import f1_m, load_model
import time

MAX_ROW = 15000

# Instanciate your FastAPI app
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class PredictionInput(BaseModel):
    dataset_path: Optional[str] = "data/predict/X_test_update.csv"
    images_path: Optional[str] = "data/predict/image_test"
    prediction_path: Optional[str] = "data/predict/predictions.csv"
    api_secured: Optional[bool] = False

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
        X["description"] = X["designation"] + " " + str(X["description"])
        
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

# Endpoint pour l'initialisation
@app.get("/initialisation")
def initialisation():
    global predictor, tokenizer, rnn, vgg16, best_weights, mapper
    
    # Charger les configurations et modèles
    with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    rnn = load_model( "models" , "best_rnn_model.h5" )
    vgg16 = load_model("models" , "best_vgg16_model.h5")

    with open("models/best_weights.json", "r") as json_file:
        best_weights = json.load(json_file)

    with open("models/mapper.json", "r") as json_file:
        mapper = json.load(json_file)

    return {"message": "Initialisation effectuée avec succès"}

# Endpoint pour la prédiction
@app.post("/prediction")
def prediction(input_data: PredictionInput, token: Optional[str] = Depends(oauth2_scheme)):
    global predictor, tokenizer, rnn, vgg16, best_weights, mapper
    
    print("token=",token)
    # Si api_secured est True, vérifiez les crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api_oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName']+" "+user_data['LastName']
            if user_data['Authorization'] < 1:
                message_response = {"message": f"{user_info} n'est pas autorisé a effectuer une prediction"}
                return message_response
    else:
        user_info = "un utilisateur inconnu"

    # Exécutez la prédiction
    t_debut = time.time()
    predictor = Predict(
        tokenizer=tokenizer,
        rnn=rnn,
        vgg16=vgg16,
        best_weights=best_weights,
        mapper=mapper,
        filepath=input_data.dataset_path,
        imagepath=input_data.images_path
    )
    predictions = predictor.predict()
    t_fin = time.time()
    
    # Sauvegarde des prédictions
    #with open(input_data.prediction_path, "w", encoding="utf-8") as json_file:
    #    json.dump(predictions, json_file, indent=2)
    predictions.to_csv(input_data.prediction_path, index=False)
        
    print("Durée de la prédiction : {:.2f}".format(t_fin - t_debut))
    
    prediction_response = {"message": f"Prédiction effectuée avec succès, demandée par {user_info}","duration": t_fin - t_debut}
    return prediction_response
