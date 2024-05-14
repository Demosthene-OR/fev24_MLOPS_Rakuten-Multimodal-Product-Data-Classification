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

class NewProductsInput(BaseModel):
    new_products_origin_folder_path: Optional[str] = "data/predict"
    new_products_dest_path: Optional[str] = "data/preprocessed"
    api_secured: Optional[bool] = False
 
@app.get("/add_new_products")
def  add_new_products(input_data: NewProductsInput, token: Optional[str] = Depends(oauth2_scheme)):
    
    # Si api_secured est True, vérifiez les crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api_oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName']+" "+user_data['LastName']
            if user_data['Authorization'] < 1:
                message_response = {"message": f"{user_info} n'est pas autorisé à ajouter de nouveaux produits"}
                return message_response
    else:
        user_info = "un utilisateur inconnu" 
        
    # Paths for the files
    origin_path = input_data.new_products_origin_folder_path
    dest_path = input_data.new_products_dest_path
    
    # File paths
    X_new_path = f"{origin_path}/X_test_update.csv"
    X_train_path = f"{dest_path}/X_train_update.csv"
    new_classes_origin_path = f"{origin_path}/new_classes.csv"
    new_classes_dest_path = f"{dest_path}/new_classes.csv"
    y_train_path = f"{dest_path}/Y_train_CVw08PX.csv"

    # Load the data
    try:
        # Append X_test_update.csv to X_train_update.csv
        X_new_df = pd.read_csv(X_new_path)
        X_train_df = pd.read_csv(X_train_path)
        X_train_combined_df = pd.concat([X_train_df, X_new_df])
        X_train_combined_df.to_csv(X_train_path, index=False)
        
        # Append new_classes.csv to new_classes.csv in preprocessed
        new_classes_origin_df = pd.read_csv(new_classes_origin_path)
        new_classes_dest_df = pd.read_csv(new_classes_dest_path)
        new_classes_combined_df = pd.concat([new_classes_dest_df, new_classes_origin_df])
        new_classes_combined_df.to_csv(new_classes_dest_path, index=False)
        
        # Update Y_train_CVw08PX.csv with cat_real column from new_classes.csv
        y_train_df = pd.read_csv(y_train_path)
        cat_real_series = new_classes_origin_df['cat_real']
        
        # Creating a new dataframe to append
        new_rows = pd.DataFrame({
            y_train_df.columns[0]: range(y_train_df.shape[0], y_train_df.shape[0] + cat_real_series.shape[0]),
            y_train_df.columns[1]: cat_real_series
        })
        
        y_train_cv_combined_df = pd.concat([y_train_df, new_rows])
        y_train_cv_combined_df.to_csv(y_train_path, index=False)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement des fichiers: {e}")
    
    return {"message": f"Les nouveaux produits ont été ajoutés avec succès par {user_info}"}    
    
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


        
    return
        
@app.get("/model_copy")
def  model_copy(input_data: UpdateProdDatasetInput, token: Optional[str] = Depends(oauth2_scheme)):
    
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

@app.get("/new_product_proposal")
def  new_product_proposal(input_data: UpdateProdDatasetInput, token: Optional[str] = Depends(oauth2_scheme)):
    
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