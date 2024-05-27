import streamlit as st
from PIL import Image
import requests
import json

title = "Rakuten - Multimodal Product Data Classification"
sidebar_name = "Introduction"
prePath = st.session_state.PrePath

      
def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    # st.image("assets/tough-communication.gif",use_column_width=True)

    st.write("")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
        
    st.title(title)
    st.markdown('''
                ---
                ''')
    st.header("**About this project**")
    st.markdown("""
        This project was carried out as part of Data Scientest's ML Ops training program, between February and June 2024.    
        <br> 
 
        """
    , unsafe_allow_html=True)

    st.header("**Goal**")
    st.markdown(
        """
        The main goal is 
        1.	to predict the category a new product belongs to
        2.	to train the models whith the new products when the accuracy on new product falls under a certain threshold
        
        All the application are launched through API and can be according the authorization level of the user:
        - Authorization 0 -> not authorized to sell new product, which means cannot predict or train the model
        - Authorization 1 -> authorized to sell new product, can predict but not train the model
        - Authorization 2 -> authorized to sell new product (predict) and train the model
    
        """
    , unsafe_allow_html=True)
    
    # Chemin d'accès à l'image
    image_path  ="assets/Rakuten_Streamlit_pages.jpg"
    image2_path ="assets/Rakuten_API.jpg"

    # Lire l'image
    image_process  = Image.open(image_path)
    image2_process = Image.open(image2_path)

    # Afficher l'image
    st.image(image_process, caption='Streamlit Pages', use_column_width=True)
    st.write("")
    st.image(image2_process, caption='API', use_column_width=True)
    
    if st.button('Click to reset Dataset in Production'):
        response = requests.get(
                'http://'+st.session_state.api_flows+':8003/reset_dataset',
                headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {st.session_state.token}"},
                data=json.dumps({
                    "api_secured": True
                })
            )
        if response.status_code == 200:
            st.success(response.json().get("message", "No message in response"))
        else:
            st.error("Failed to reset Dataset")
