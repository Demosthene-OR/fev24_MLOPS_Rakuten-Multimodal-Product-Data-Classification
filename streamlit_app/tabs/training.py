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



title = "Model Training"
sidebar_name = "Model Training"
prePath = st.session_state.PrePath

def display_header():
    st.write("")
    st.title(title)
    st.markdown('''
                ---
                ''')
def run():
    display_header()
    if st.session_state.UserAuthorization <1:
        st.write("You are not logged in or not authorized to sell product !")
        return

    chosen_id = tab_bar(data=[
        TabBarItemData(id="tab1", title="Accuracy", description="of prediction"),
        TabBarItemData(id="tab2", title="Train the model", description="with latest sales")],
        default="tab1")

    if (chosen_id == "tab1"):
        num_sales = st.number_input("Number of sales to measure accuracy since last full train:", min_value=1, max_value=1000, value=10)
        response = requests.get(
            'http://'+st.session_state.api_flows+':8003/compute_metrics',
            headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {st.session_state.token}"},
            data=json.dumps({
                "classes_path": "data/preprocessed/new_classes.csv",
                "api_secured": True
                }))
        accuracy = response.json().get("accuracy", None)
        st.write("#### Accuracy on the last",num_sales,"sales = ",accuracy)
    if (chosen_id == "tab2"):
        # Chemin du dossier principal
        model_folder = st.session_state.PrePath+'models'

        # Obtenir la liste des sous-dossiers
        subfolders = [f.name for f in os.scandir(model_folder) if f.is_dir()]
        subfolders.insert(0, 'Production')
        
        # Sous-dossiers à supprimer
        folders_to_remove = ['best_rnn_model', 'best_vgg16_model']

        # Supprimer les sous-dossiers spécifiés de la liste
        subfolders = [folder for folder in subfolders if folder not in folders_to_remove]
        st.write("List of saved models you can use to initialize the training:")
        st.write(subfolders)
        sel_model = st.selectbox("Model selected:",subfolders) # label_visibility="hidden")
        if sel_model == "Production":
            model_dir = ""
        else:
            model_dir = "/"+sel_model
                
        if st.button('Click to start the training process'):
            response = requests.get(
                'http://'+st.session_state.api_flows+':8003/save_model_start_train',
                headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {st.session_state.token}"},
                data=json.dumps({
                    "model_path": "models"+model_dir,
                    "dataset_path":"data/preprocessed",
                    "api_secured": True
                    }))
            if response.status_code == 200:
                st.success("Training processed over !")
                st.success(response.json().get("message", "No message in response"))
            else:
                st.error("Failed to train the model")
                st.error(response.json().get("message", "No message in response"))

        
# if __name__ == "__main__":
#     run()
