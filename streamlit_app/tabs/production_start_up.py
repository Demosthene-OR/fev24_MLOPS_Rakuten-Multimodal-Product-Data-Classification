import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from fastapi import FastAPI, Depends, HTTPException, status
import os
import json
from datetime import time
from extra_streamlit_components import tab_bar, TabBarItemData
import shutil



title = "Production Release"
sidebar_name = "Production Release"
prePath = st.session_state.PrePath

def display_header():
    st.write("")
    st.title(title)
    st.markdown('''
                ---
                ''')
def run():   
    st.write('Hello')
    display_header()
    if st.session_state.UserAuthorization <2:
        st.write("You are not logged in or not authorized to sell product !")
        return
    
    
    # Chemin du dossier principal
    model_folder = st.session_state.PrePath+'models'

    # Obtenir la liste des sous-dossiers
    subfolders = [f.name for f in os.scandir(model_folder) if f.is_dir()]
        
    # Sous-dossiers à supprimer
    folders_to_remove = ['best_rnn_model', 'best_vgg16_model']

    # Supprimer les sous-dossiers spécifiés de la liste
    subfolders = [folder for folder in subfolders if folder not in folders_to_remove]
    
    chosen_id = tab_bar(data=[
    TabBarItemData(id="tab1", title="Models Performance", description=""),
    TabBarItemData(id="tab2", title="Release a model version", description="")],
    default="tab1")
    
    if (chosen_id == "tab1"):

        st.write("Select the model:")
        sel_model = st.selectbox("Model selected:",subfolders) # label_visibility="hidden")
        
        # Chemin vers le fichier performances.json
        file_path = prePath+"models/"+sel_model+'/performances.json'  

        # Vérifier si le fichier existe
        if os.path.exists(file_path):
            # Ouvrir et lire le fichier JSON
            with open(file_path, 'r') as file:
                json_data = json.load(file)
    
            # Convertir le dictionnaire en une chaîne formatée
            formatted_json_string = json.dumps(json_data, indent=4)
            st.code(formatted_json_string, language='json')

    if (chosen_id == "tab2"): 
        st.write("Select the model:")
        sel_model = st.selectbox("Model selected:",subfolders) # label_visibility="hidden")
        # Chemin vers le fichier performances.json
        files_path = prePath+"models/"+sel_model  
        
        # Define the files and directories to copy
        items_to_copy = [
            "best_weights.json",
            "mapper.json",
            "tokenizer_config.json",
            "best_rnn_model",
            "best_vgg16_model"
        ]
        
        # Destination path
        dest_path = os.path.join(prePath, "models")

        # Function to copy files and directories
        def copy_items(src_folder, dest_folder, items):
            for item in items:
                src_path = os.path.join(src_folder, item)
                dest_path = os.path.join(dest_folder, item)

                try:
                    if os.path.isfile(src_path):
                        shutil.copy(src_path, dest_path)
                        st.success(f"Copied file: {item}")
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                        st.success(f"Copied directory: {item}")
                except Exception as e:
                    st.error(f"Error copying {item}: {e}")

        # Call the function to copy items
        if st.button('Click to release the model'):
            copy_items(files_path, dest_path, items_to_copy)
