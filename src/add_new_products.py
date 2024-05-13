from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional
import requests

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

# Instanciate your FastAPI app
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class ComputeMetricsInput(BaseModel):
    pending_classes_path: Optional[str] = "data/predict/new_classes.csv"
    api_secured: Optional[bool] = False
    
class UpdatePendingProductsInput(BaseModel):
    new_dataset_path: Optional[str] = "data/predict/X_test_update.csv"
    new_images_path: Optional[str] = "data/predict/image_test"
    new_classes_path: Optional[str] = "data/predict/new_classes.csv"
    pending_new_dataset_path: Optional[str] = "data/pending/X_train.csv"
    pending_new_images_path: Optional[str] = "data/pending/image"
    pending_new_y_path: Optional[str] = "data/pending/Y_train.csv"
    pending_classes_path: Optional[str] = "data/predict/new_classes.csv"
    api_secured: Optional[bool] = False

class UpdateProdDatasetInput(BaseModel):
    pending_new_dataset_path: Optional[str] = "data/pending/X_train.csv"
    pending_new_images_path: Optional[str] = "data/pending/image"
    pending_new_y_path: Optional[str] = "data/pending/Y_train.csv"
    pending_classes_path: Optional[str] = "data/predict/new_classes.csv"
    dataset_path: Optional[str] = "data/preprocessed/X_train_update.csv"
    images_path: Optional[str] = "data/preprocessed/image_train"
    y_path: Optional[str] = "data/preprocessed/Y_train_CVw08PX.csv"
    api_secured: Optional[bool] = False
    
@app.get("/compute_metrics_pending")
def compute_metrics_pending(input_data: ComputeMetricsInput, token: Optional[str] = Depends(oauth2_scheme)):
    
    # Si api_secured est True, vérifiez les crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api_oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName']+" "+user_data['LastName']
            if user_data['Authorization'] < 1:
                message_response = {"message": f"{user_info} n'est pas autorisé à calculer les metrics des nouveaux produits en attente"}
                return message_response
    else:
        user_info = "un utilisateur inconnu"
    
    return

@app.get("/add_new_products_to_pending")
def  update_pending_products(input_data: UpdatePendingProductsInput, token: Optional[str] = Depends(oauth2_scheme)):
    
    # Si api_secured est True, vérifiez les crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api_oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName']+" "+user_data['LastName']
            if user_data['Authorization'] < 1:
                message_response = {"message": f"{user_info} n'est pas autorisé à ajouter de nouveaux produits en attente"}
                return message_response
    else:
        user_info = "un utilisateur inconnu"
        
    return
        
@app.get("/add_new_products_to_prod")
def  update_prod_products(input_data: UpdateProdDatasetInput, token: Optional[str] = Depends(oauth2_scheme)):
    
    # Si api_secured est True, vérifiez les crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api_oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName']+" "+user_data['LastName']
            if user_data['Authorization'] < 1:
                message_response = {"message": f"{user_info} n'est pas autorisé à ajouter de nouveaux produits en production"}
                return message_response
    else:
        user_info = "un utilisateur inconnu"
        
    return

# if __name__ == "__main__":
#     main()