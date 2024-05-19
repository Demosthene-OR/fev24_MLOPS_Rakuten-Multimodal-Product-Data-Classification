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
        
        
# if __name__ == "__main__":
#     run()
