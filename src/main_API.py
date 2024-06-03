from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional
import requests
import asyncio

from src.features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from src.models.train_model_API import TextRnnModel, ImageVGG16Model, concatenate
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import tensorflow as tf
import sys
import json
import time
import numpy as np
import pandas as pd
from src.tools import f1_m, load_model
import sys
import random
import datetime

# Instanciate your FastAPI app
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TrainInput(BaseModel):
    x_train_path: Optional[str] = "data/preprocessed/X_train_update.csv"
    y_train_path: Optional[str] = "data/preprocessed/Y_train_CVw08PX.csv"
    images_path: Optional[str] = "data/preprocessed/image_train"
    model_path: Optional[str] = "models"
    n_epochs: Optional[int] = 1
    samples_per_class: Optional[int] = 50  # Caution: If samples_per_class==0 , the Full Dataset will be used
    with_test: Optional[bool] = False
    random_state: Optional[int] = 42
    full_train: Optional[bool] = True
    n_sales_ft: Optional[int] = 50
    api_secured: Optional[bool] = False

@app.post("/train")
async def main(input_data: TrainInput, token: Optional[str] = Depends(oauth2_scheme)):
    
    # Si api_secured est True, vérifiez les crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api_oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à l'entrainment du modèle")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName']+" "+user_data['LastName']
            if user_data['Authorization'] < 2:
                prediction_response = {"message": f"{user_info} n'est pas autorisé à effectuer l'entrainment du modèle"}
                return prediction_response
    else:
        user_info = "un utilisateur inconnu"
    
    with_test = False if input_data.with_test==0 else True
    samples_per_class = input_data.samples_per_class
    n_epochs = input_data.n_epochs
    full_train = input_data.full_train
    n_sales_ft = input_data.n_sales_ft
    random_state = input_data.random_state if input_data.random_state >= 0 else random.randint(0, 100)

    t_debut = time.time()
    data_importer = DataImporter(input_data.x_train_path,input_data.y_train_path, input_data.model_path )
    df = data_importer.load_data()
    
    if full_train:
        X_train, X_val, X_test, y_train, y_val, y_test = \
            data_importer.split_train_test(df, samples_per_class=samples_per_class, random_state=random_state, with_test=with_test) 
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = \
            data_importer.split_train_test(df, samples_per_class=10, random_state=random_state, with_test=with_test) 
        df2 = df[-n_sales_ft:]
        y_train2 = df2["prdtypecode"]
        X_train2 = df2.drop(["prdtypecode"], axis=1)
        y_train = pd.concat([y_train,y_train2], axis=0)
        X_train = pd.concat([X_train,X_train2], axis=0)
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        print('============================')
        print("Final Finetuning Dataset size : ", len(X_train)+len(X_val)+len(X_test))
        print("Final Finetuning Train size   : ", len(X_train))
        print("Final Finetuning Val size     : ", len(X_val))
        print("Final Finetuning Test size    : ", len(X_test))
        print('============================')
        samples_per_class = 0    

    # Preprocess text and images
    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor(input_data.images_path)
    text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
    text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
    image_preprocessor.preprocess_images_in_df(X_train)
    image_preprocessor.preprocess_images_in_df(X_val)
    if with_test:
        text_preprocessor.preprocess_text_in_df(X_test, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X_test)

    # sys.exit(0)
    # Train Rnn model
    print('============================')
    print("Training RNN Model")
    text_rnn_model = TextRnnModel(file_path=input_data.model_path)
    rnn_history, rnn_best_epoch, rnn_best_f1, rnn_best_accuracy = text_rnn_model.preprocess_and_fit(X_train, y_train, X_val, y_val, n_epochs=n_epochs, full_train=full_train)
    print("Finished training RNN")
    
    print('============================')
    print("Training VGG")
    # Train VGG16 model
    image_vgg16_model = ImageVGG16Model(file_path=input_data.model_path)
    vgg16_history, vgg16_best_epoch, vgg16_best_f1, vgg16_best_accuracy = image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val, n_epochs=n_epochs, full_train=full_train)
    print("Finished training VGG")
    
    print('============================')
    with open(input_data.model_path+"/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    rnn = load_model(input_data.model_path,"best_rnn_model.h5")
    vgg16 = load_model(input_data.model_path,"best_vgg16_model.h5")
           
    print("Training the concatenate model")
    model_concatenate = concatenate(tokenizer, rnn, vgg16)
    if (samples_per_class > 0):
        new_samples_per_class = min(samples_per_class,50)  # max(int(samples_per_class/12),3) # 50
    else:
        if full_train:
            new_samples_per_class = 50
        else:
            new_samples_per_class = 0

    rnn_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train, new_samples_per_class=new_samples_per_class, random_state=random_state)  
    best_weights, best_weighted_f1, best_accuracy, concatenate_train_size = model_concatenate.optimize(rnn_proba, vgg16_proba, new_y_train)

    with open(input_data.model_path+"/best_weights.json", "w") as file:
        json.dump(best_weights, file)
          
    t_fin = time.time()
    training_duration = t_fin - t_debut
    print("Training duration = {:.2f} sec".format(training_duration))
    print("Finished training concatenate model")
    print('============================')
    
    # Enregistre le modèle au format h5
    # concatenate_model.save(input_data.model_path+"/concatenate.h5")
    
    # Calcul de la perforance sur le dataset test
    t_debut = time.time()
    t_fin = t_debut
    concatenate_test_size = 0
    test_accuracy = 0
    test_f1 = 0
    if with_test:
        rnn_proba_test, vgg16_proba_test, new_y_test = model_concatenate.predict(X_test, y_test, new_samples_per_class=0, random_state=random_state) 
        combined_predictions = (best_weights[0] * rnn_proba_test) + (best_weights[1] * vgg16_proba_test)
        final_predictions = np.argmax(combined_predictions, axis=1)
        concatenate_test_size = len(new_y_test)
        test_accuracy = accuracy_score(new_y_test, final_predictions)
        test_f1 = f1_score(new_y_test , final_predictions, average='weighted')
        t_fin = time.time()
        print('============================')
        print("Testing the concatenate model")
        print("Test dataset size :", concatenate_test_size)
        print("Test: f1 score =", test_f1)
        print("Test accuracy score =", test_accuracy)
        print("Test duration = {:.2f} sec".format(t_fin - t_debut))
        print('============================')
    
    test_duration = t_fin - t_debut
    train_size = int(len(X_train))
    val_size = int(len(X_val))
    test_size = int(len(X_test))
    performances_recap = {
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Input": {
            "epochs_requested": int(n_epochs),
            "samples_per_class": int(samples_per_class),
            "with_test": int(with_test),
            "random_state": int(random_state),
            "Dataset_size": {
                "Train": train_size,
                "Val": val_size,
                "Test": test_size
                }    
        },
        "Text" : {
            "best_epoch": int(rnn_best_epoch+1), 
            "f1": float(rnn_best_f1),
            "accuracy" : float(rnn_best_accuracy)
        },
        "VGG16" : {
            "best_epoch": int(vgg16_best_epoch+1), 
            "f1": float(vgg16_best_f1),
            "accuracy" : float(vgg16_best_accuracy)
        },
        "Concatenate" : {
            "weight": best_weights,
            "Train": {
                "f1": float(best_weighted_f1),
                "accuracy": float(best_accuracy),
                "duration" : int(training_duration),
                "size": int(concatenate_train_size)
            },
            "Test": {
                "f1": float(test_f1),
                "accuracy": float(test_accuracy),
                "duration" : int(test_duration),
                "size": concatenate_test_size
            }
        }
    }
    with open(input_data.model_path+"/performances.json", "w") as file:
        json.dump( performances_recap, file, indent=4)
        
    return {"message": "Entrainement effectuée avec succès"}
    