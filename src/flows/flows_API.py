from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional
import requests

import pandas as pd
import numpy as np
import json
import time
import os
import shutil
import glob
from datetime import datetime

# Instanciate your FastAPI app
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class NewProductsProposalInput(BaseModel):
    num_products: Optional[int] = 3
    new_products_folder_path: Optional[str] = "data/predict"
    api_secured: Optional[bool] = False

class NewProductsInput(BaseModel):
    new_products_origin_path: Optional[str] = "data/predict"
    new_products_dest_path: Optional[str] = "data/preprocessed"
    api_secured: Optional[bool] = False

class ComputeMetricsInput(BaseModel):
    classes_path: Optional[str] = "data/preprocessed/new_classes.csv"
    num_sales: Optional[int] = 10
    api_secured: Optional[bool] = False

class SaveModelTrain(BaseModel):
    model_path: Optional[str] = "models"
    dataset_path: Optional[str] = "data/preprocessed"
    n_epochs:Optional[int] = 1
    samples_per_class: Optional[int] = 0
    full_train: Optional[bool] = True
    n_sales_ft: Optional[int] = 50
    api_secured: Optional[bool] = False
    

@app.get("/new_product_proposal")
def new_product_proposal(input_data: NewProductsProposalInput, token: Optional[str] = Depends(oauth2_scheme)):
    
    # If api_secured = True, check the crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api-oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName'] + " " + user_data['LastName']
            if user_data['Authorization'] < 1:
                message_response = {"message": f"{user_info} n'est pas autorisé à ajouter de nouveaux produits en production"}
                return message_response
    else:
        user_info = "un utilisateur inconnu"

    num_products = input_data.num_products
    
    # Paths for the files
    folder_path = input_data.new_products_folder_path
    reserve_path = f"{folder_path}/new_products_reserve/X_test_update.csv"
    proposal_path = f"{folder_path}/X_test_update.csv"

    # Load the data
    try:
        if os.path.exists(reserve_path):
            reserve_df = pd.read_csv(reserve_path, index_col=0)
            if len(reserve_df) < num_products:
                raise HTTPException(status_code=400, detail="Nombre de produits disponibles insuffisant dans le fichier de réserve.")

            # Get the specified number of products from the reserve
            proposed_products_df = reserve_df.head(num_products)
            # proposed_products_df = proposed_products_df.rename(columns={'Unnamed: 0': ''})
            
            # Save the proposed products to the new file
            proposed_products_df.to_csv(proposal_path)
                    
            # Parcourir le DataFrame et copier chaque image
            base_source_path = folder_path+"/new_products_reserve/image_test"
            base_destination_path = folder_path+"/image_test"
            for index, row in proposed_products_df.iterrows():
                imageid = row['imageid']
                productid = row['productid']
                image_name = f"image_{imageid}_product_{productid}.jpg"
                source_path = os.path.join(base_source_path, image_name)
                destination_path = os.path.join(base_destination_path, image_name)
                
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                
            # Previous prediction deletion (if remained)
            file_path = os.path.join(folder_path, "new_classes.csv")
            if os.path.exists(file_path):
                new_classes_df = pd.read_csv(f"{folder_path}/new_classes.csv")
                new_classes_df = new_classes_df.drop(new_classes_df.index)
                new_classes_df.to_csv(f"{folder_path}/new_classes.csv", index=False)
            files_to_delete = ["predictions.csv"]
            for file_name in files_to_delete:
                file_path = os.path.join(folder_path, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)

            return {"message": f"{num_products} produits ont été proposés par {user_info}"}
        else:
            raise HTTPException(status_code=404, detail="Le fichier de réserve des nouveaux produits est introuvable.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement des fichiers: {e}")


@app.get("/add_new_products")
def  add_new_products(input_data: NewProductsInput, token: Optional[str] = Depends(oauth2_scheme)):
    
    # If api_secured = True, check the crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api-oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
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
    origin_path = input_data.new_products_origin_path
    dest_path = input_data.new_products_dest_path
    
    # File paths
    X_new_path = f"{origin_path}/X_test_update.csv"
    X_train_path = f"{dest_path}/X_train_update.csv"
    new_classes_origin_path = f"{origin_path}/new_classes.csv"
    new_classes_dest_path = f"{dest_path}/new_classes.csv"
    y_train_path = f"{dest_path}/Y_train_CVw08PX.csv"   
    image_test_path = f"{origin_path}/image_test"
    image_train_path = f"{dest_path}/image_train"
    reserve_path = f"{origin_path}/new_products_reserve/X_test_update.csv"

    try:
        # Append X_test_update.csv to X_train_update.csv
        X_new_df = pd.read_csv(X_new_path, index_col=0)
        X_train_df = pd.read_csv(X_train_path, index_col=0)
        
        X_train_combined_df = pd.concat([X_train_df, X_new_df]) # , ignore_index=False)

        # X_train_combined_df = X_train_combined_df.astype({
        #     'productid': int,
        #     'imageid': int
        # }) 
        X_train_combined_df.to_csv(X_train_path)
        
        # Append new_classes.csv to new_classes.csv in preprocessed
        new_classes_origin_df = pd.read_csv(new_classes_origin_path)
        new_classes_dest_df = pd.read_csv(new_classes_dest_path)
        new_classes_combined_df = pd.concat([new_classes_dest_df, new_classes_origin_df])
        new_classes_combined_df.to_csv(new_classes_dest_path, index=False)
        
        # Update Y_train_CVw08PX.csv with cat_real column from new_classes.csv
        y_train_df = pd.read_csv(y_train_path, index_col=0)
        y_train_df['prdtypecode'].astype(int)
        cat_real_series = new_classes_origin_df['cat_real'].astype(int)
        
        # Creating a new dataframe to append
        new_rows = pd.DataFrame({'prdtypecode': cat_real_series})
        new_rows.index = X_new_df.index
        
        y_train_cv_combined_df = pd.concat([y_train_df, new_rows], ignore_index=False)
        y_train_cv_combined_df.to_csv(y_train_path)

        # Copy images from image_test to image_train
        if not os.path.exists(image_train_path):
            os.makedirs(image_train_path)
        
        for filename in os.listdir(image_test_path):
            src_file = os.path.join(image_test_path, filename)
            dest_file = os.path.join(image_train_path, filename)
            shutil.copy(src_file, dest_file)
            
        # Remove lines from new_products_reserve/X_test_update.csv that are in X_test_update.csv
        if os.path.exists(reserve_path):
            reserve_df = pd.read_csv(reserve_path, index_col=0)
            reserve_df = reserve_df.iloc[len(new_classes_origin_df):]
            reserve_df.to_csv(reserve_path)
            
        # Delete or empty specific files in data/predict
        new_classes_origin_df = new_classes_origin_df.drop(new_classes_origin_df.index)
        new_classes_origin_df.to_csv(new_classes_origin_path, index=False)
        X_new_df = X_new_df.drop(X_new_df.index)
        X_new_df.to_csv(X_new_path)
        files_to_delete = ["predictions.csv"]
        for file_name in files_to_delete:
            file_path = os.path.join(origin_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Delete all files in data/predict/image_test
        if os.path.exists(image_test_path):
            pattern = os.path.join(image_test_path, "image_*")
            image_files = glob.glob(pattern)
        # Parcourir tous les fichiers dans le répertoire et les supprimer
            for filename in image_files: # os.listdir(image_test_path):
                file_path = os.path.join(image_test_path, filename)
                if os.path.isfile(file_path): # or os.path.islink(file_path):
                    os.unlink(file_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement des fichiers: {e}")
    
    return {"message": f"Les nouveaux produits ont été ajoutés avec succès par {user_info}"}    


@app.get("/compute_metrics")
def compute_metrics_new_products(input_data: ComputeMetricsInput, token: Optional[str] = Depends(oauth2_scheme)):

    # If api_secured = True, check the crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api-oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName'] + " " + user_data['LastName']
            if user_data['Authorization'] < 2:
                message_response = {"message": f"{user_info} n'est pas autorisé à calculer les metrics des nouveaux produits en attente"}
                return message_response
    else:
        user_info = "un utilisateur inconnu"
        
    try:
        # Chemin du fichier
        new_classes_path = input_data.classes_path
        
        # Chargement des données
        new_classes_df = pd.read_csv(new_classes_path)
        
        # Calcul de l'accuracy
        num_new_products = len(new_classes_df)
        if num_new_products>0:
            num_new_products = min(num_new_products, input_data.num_sales)
            accuracy = (new_classes_df['cat_real'].iloc[-num_new_products:] == new_classes_df['cat_pred'].iloc[-num_new_products:]).mean()
        else:
            accuracy = 1.0
        
        return {"accuracy": accuracy, "num_sales":num_new_products}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul des metrics: {e}")


@app.get("/save_model_start_train")
def save_model_start_train(input_data: SaveModelTrain, token: Optional[str] = Depends(oauth2_scheme)):
    
    # If api_secured = True, check the crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api-oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName'] + " " + user_data['LastName']
            if user_data['Authorization'] < 2:
                message_response = {"message": f"{user_info} n'est pas autorisé à ajouter de nouveaux produits en production"}
                return message_response
    else:
        user_info = "un utilisateur inconnu"
        
    try:
        # Chemins des répertoires
        source_dir = input_data.model_path
        current_time = datetime.now().strftime("%Y-%m-%d %H%M")
        if input_data.full_train:
            destination_dir = f"models/saved_models - {current_time} - Full"
        else:
            destination_dir = f"models/saved_models - {current_time} - Fine-tuned"

        # Liste des fichiers et répertoires à copier
        items_to_copy = [
            "best_weights.json",
            "mapper.json",
            "tokenizer_config.json",
            "best_rnn_model",
            "best_vgg16_model",
            "best_rnn_model.h5",
            "best_vgg16_model.h5"
        ]

        # Création du répertoire de destination
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)

        # Copie des fichiers et répertoires
        for item in items_to_copy:
            source_path = os.path.join(source_dir, item)
            destination_path = os.path.join(destination_dir, item)
            if os.path.exists(source_path):
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, destination_path)
                else:
                    shutil.copy2(source_path, destination_path)             
        
        # Copie du dataset pour ne pas être déranger
        items_to_copy = [
            "X_train_update.csv",
            "Y_train_CVw08PX.csv",
        ] 
        source_dir = input_data.dataset_path
        for item in items_to_copy:
            source_path = os.path.join(source_dir, item)
            destination_path = os.path.join(destination_dir, item)
            shutil.copy2(source_path, destination_path)
            
        # Vérifier si new_classes.csv existe et le rendre vide si c'est le cas
        new_classes_path = input_data.dataset_path + "/new_classes.csv"
        if os.path.exists(new_classes_path):
            try:
                df = pd.read_csv(new_classes_path)
                columns = df.columns
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(new_classes_path, index=False)
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {new_classes_path}: {e}")
                
        # Exécution de la requête POST pour démarrer l'entraînement
        train_endpoint = "http://api-train:8002/train"
        train_data = {
            "api_secured": True,
            "x_train_path": destination_dir+"/X_train_update.csv",
            "y_train_path":destination_dir+"/Y_train_CVw08PX.csv",
            "images_path": "data/preprocessed/image_train",
            "model_path": destination_dir, 
            "n_epochs": input_data.n_epochs, 
            "samples_per_class": input_data.samples_per_class,
            "full_train": input_data.full_train,
            "n_sales_ft": input_data.n_sales_ft,
            "with_test": True
        }
        train_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        train_response = requests.post(train_endpoint, json=train_data, headers=train_headers)
        
        if train_response.status_code == 200:          
            return {"message": f"Le modèle {destination_dir} a été sauvegardé avec succès par {user_info} et l'entraînement s'est correctement terminé"}
        else:
            raise HTTPException(status_code=train_response.status_code, detail=f"Erreur lors de la sauvegarde du model dans {destination_dir} \
                ou lors de la requête d'entraînement: {train_response.text}")


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde du modèle: {e}")

@app.get("/reset_dataset")
def  reset_dataset(input_data: NewProductsInput, images: Optional[bool] = True, token: Optional[str] = Depends(oauth2_scheme)):
    
    # If api_secured = True, check the crédentiels
    if input_data.api_secured:
        auth_response = requests.get("http://api-oauth:8001/secured", headers={"Authorization": f"Bearer {token}"})
        if auth_response.status_code != 200:
            raise HTTPException(status_code=auth_response.status_code, detail="Non autorisé à accéder à la prédiction")
        else:
            user_data = auth_response.json()
            user_info = user_data['FirstName'] + " " + user_data['LastName']
            if user_data['Authorization'] < 2:
                message_response = {"message": f"{user_info} n'est pas autorisé à ajouter de nouveaux produits en production"}
                return message_response
    else:
        user_info = "un utilisateur inconnu"
    
    num_train = 84916
    
    # Paths for the files
    origin_path = input_data.new_products_origin_path
    dest_path = input_data.new_products_dest_path
    
    # File paths
    X_new_path = f"{origin_path}/X_test_update.csv"
    X_train_path = f"{dest_path}/X_train_update.csv"
    X_test_path = f"{dest_path}/X_test_update.csv"
    new_classes_origin_path = f"{origin_path}/new_classes.csv"
    new_classes_dest_path = f"{dest_path}/new_classes.csv"
    y_train_path = f"{dest_path}/Y_train_CVw08PX.csv"   
    image_train_path = f"{dest_path}/image_train"
    reserve_path = f"{origin_path}/new_products_reserve/X_test_update.csv"
    
    try:
        # Lecture des fichiers à réinitialiser
        X_new_df = pd.read_csv(X_new_path, index_col=0)
        X_train_df = pd.read_csv(X_train_path, index_col=0)
        new_classes_origin_df = pd.read_csv(new_classes_origin_path)
        new_classes_dest_df = pd.read_csv(new_classes_dest_path)
        y_train_df = pd.read_csv(y_train_path, index_col=0)
        
        if images:
            # Parcourir le DataFrame X_train de preprocessed et supprimer les images ajoutées
            if len(X_train_df) > num_train:
                for index, row in X_train_df.iloc[num_train:].iterrows():
                        imageid = row['imageid']
                        productid = row['productid']
                        image_name = f"image_{imageid}_product_{productid}.jpg"
                        file_path = os.path.join(image_train_path, image_name)
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path) 
        
        # Effacement des image dans origin_path
        pattern = os.path.join(origin_path+"/image_test", "image_*")
        image_files = glob.glob(pattern)
        for file in image_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")

        
        # Réinitialisation
        X_new_df = X_new_df.drop(X_new_df.index)
        # X_train_df = X_train_df.iloc[:num_train]
        # y_train_df = y_train_df.iloc[:num_train]
        new_classes_origin_df = new_classes_origin_df.drop(new_classes_origin_df.index)
        new_classes_dest_df = new_classes_dest_df.drop(new_classes_dest_df.index)
        shutil.copy(X_test_path, reserve_path)
        
        # Ecriture des Datasets réinitialisés
        X_new_df.to_csv(X_new_path)
        shutil.copy(f"{dest_path}/X_train_update - saved.csv" , X_train_path)
        shutil.copy(f"{dest_path}/Y_train_CVw08PX - saved.csv" , y_train_path)
        # X_train_df.to_csv(X_train_path)
        # y_train_df.to_csv(y_train_path)
        new_classes_origin_df.to_csv(new_classes_origin_path, index=False)
        new_classes_dest_df.to_csv(new_classes_dest_path, index=False)
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la réinitialisation des fichiers: {e}")
    
    return {"message": f"Les datasets ont été réinitalisés par {user_info}"}   
    

# if __name__ == "__main__":
#     main()
