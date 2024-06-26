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
    if st.session_state.UserAuthorization <2:
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
                "num_sales": num_sales,
                "api_secured": True
                }))
        accuracy = response.json().get("accuracy", None)
        num_sales = response.json().get("num_sales", None)
        st.write("#### Accuracy on the last",num_sales,"sales = ",accuracy)
    if (chosen_id == "tab2"):
        
        n_epochs = st.number_input("Maximum number of epoch(s)                        :", min_value=1, max_value=50, value=1)
        fine_tune = st.checkbox("Do you want to Fine-tune the model ?", value=False)
        full_train = not fine_tune
        if fine_tune:
            n_sales_ft = st.number_input("Number of last sales to take into account for the finetuning:", min_value=50, max_value=300, value=150)
            samples_per_class = 0
        else:
            n_sales_ft = 50
            samples_per_class = st.number_input("Number of samples per class (0 = Full Training set):", min_value=0, max_value=600, value=0)
            
        # Chemin du dossier principal
        model_folder = st.session_state.PrePath+'models'

        # Obtenir la liste des sous-dossiers
        subfolders = [f.name for f in os.scandir(model_folder) if f.is_dir() and "saved_model" in f.name and "Full" in f.name]
        if not fine_tune:
            subfolders.insert(0, 'From scratch')
        # subfolders.insert(0, 'Production')
        
        st.write("List of saved models you can use to initialize the training:")
        st.write(subfolders)
        sel_model = st.selectbox("Model selected:",subfolders) # label_visibility="hidden")
        # if sel_model == "Production":
        #     model_dir = ""
        if sel_model == "From scratch":
            model_dir = "/empty_model"
        else:
            model_dir = "/"+sel_model
                        
        if st.button('Click to start the training process'):
            response = requests.get(
                'http://'+st.session_state.api_flows+':8003/save_model_start_train',
                headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {st.session_state.token}"},
                data=json.dumps({
                    "model_path": "models"+model_dir,
                    "dataset_path":"data/preprocessed",
                    "n_epochs": n_epochs,
                    "samples_per_class":samples_per_class,
                    "full_train": full_train,
                    "n_sales_ft": n_sales_ft,
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
