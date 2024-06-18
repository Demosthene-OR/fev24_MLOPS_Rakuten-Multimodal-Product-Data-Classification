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
import altair as alt

title = "Production Release"
sidebar_name = "Production Release"
prePath = st.session_state.PrePath

def display_header():
    st.write("")
    st.title(title)
    st.markdown('''
                ---
                ''')
def collect_preformances():
    # Obtenir la liste des sous-dossiers
    model_folder = st.session_state.PrePath+'models'
    subfolders = [f.name for f in os.scandir(model_folder) if f.is_dir() and "saved_model" in f.name]
    # Initialiser une liste pour stocker les résultats
    x_date = []
    y_train_f1 = []
    y_train_accuracy = []
    y_test_f1 = []
    y_test_accuracy = []
    y_text_weight = []

    # Parcourir chaque sous-dossier
    for subfolder in sorted(subfolders):
        # Chemin vers le fichier performances.json
        performance_file = os.path.join(model_folder, subfolder, 'performances.json')
        
        # Lire le fichier performances.json
        with open(performance_file, 'r') as file:
            data = json.load(file)
        
        # Extraire les valeurs de f1 et d'accuracy du test
        train_f1 = data["Concatenate"]["Train"]["f1"]
        train_accuracy = data["Concatenate"]["Train"]["accuracy"]
        test_f1 = data["Concatenate"]["Test"]["f1"]
        test_accuracy = data["Concatenate"]["Test"]["accuracy"]
        text_weight = data["Concatenate"]["weight"][0]
        
        # Ajouter les résultats à la liste
        x_date.append(subfolder[17:26]+subfolder[31:])
        y_train_f1.append(train_f1) 
        y_train_accuracy.append(train_accuracy)
        y_test_f1.append(test_f1)
        y_test_accuracy.append(test_accuracy) 
        y_text_weight.append(text_weight)
    return x_date, y_train_f1, y_train_accuracy, y_test_f1, y_test_accuracy,y_text_weight
    

def display_performance_graph(x_date, y_train_f1, y_train_accuracy, y_test_f1, y_test_accuracy, y_text_weight):
    # Créer un dataframe pour Altair
    data = pd.DataFrame({
        'Date': x_date,
        'Train F1 Score': y_train_f1,
        'Train Accuracy': y_train_accuracy,
        'Test F1 Score': y_test_f1,
        'Test Accuracy': y_test_accuracy,
        'Text Weight': y_text_weight
    })

    # Transformer les données pour Altair
    data_melted = data.melt('Date', var_name='Metric', value_name='Value')

    # Créer le tracé de la courbe principale avec Altair
    chart_line = alt.Chart(data_melted).mark_line(point=alt.OverlayMarkDef(filled=False, fill="white",size=80)).encode(
        x=alt.X('Date', axis=alt.Axis(labelAngle=15, title=None)),  # Spécifier 'Date' comme temps (Time) avec rotation des étiquettes
        y=alt.Y('Value:Q', scale=alt.Scale(domain=[0.65, 1.0])),  # Limiter l'échelle de l'axe Y entre 0.7 et 1.0
        color='Metric:N',  # Colorer en fonction de la métrique (Metric)
    ).properties(
        title='Concatenate Model Performances over Time',
        height=500 
        ).configure_title(
            fontSize=16,
            anchor='middle'
            )

    # Afficher le graphe dans Streamlit avec une largeur de conteneur adaptative
    st.altair_chart(chart_line.configure_axis(labelFontSize=12), use_container_width=True)
   
def run():   
    st.write('Hello')
    display_header()
    if st.session_state.UserAuthorization <2:
        st.write("You are not logged in or not authorized to sell product !")
        return
    
    
    # Chemin du dossier principal
    model_folder = st.session_state.PrePath+'models'

    # Obtenir la liste des sous-dossiers
    subfolders = [f.name for f in os.scandir(model_folder) if f.is_dir() and "saved_model" in f.name]
          
    chosen_id = tab_bar(data=[
    TabBarItemData(id="tab1", title="Models Performance", description=""),
    TabBarItemData(id="tab2", title="Release a model version", description="")],
    default="tab1")
    
    if (chosen_id == "tab1"):

        st.link_button("Go to Tensorboard to see\nText and image models metrics", "http://localhost:6006/")
        
        x_date, y_train_f1, y_train_accuracy, y_test_f1, y_test_accuracy, y_text_weight = collect_preformances()
        display_performance_graph(x_date, y_train_f1, y_train_accuracy, y_test_f1, y_test_accuracy, y_text_weight)
        
        st.write("Select the model to see its details :")
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
            "best_vgg16_model",
            "best_rnn_model.h5",
            "best_vgg16_model.h5"     
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
            predict_endpoint = "http://api-predict:8000/initialisation"
            predict_response = requests.get(predict_endpoint)
            st.success(predict_response.json().get("message", "No message in response"))
            
            
