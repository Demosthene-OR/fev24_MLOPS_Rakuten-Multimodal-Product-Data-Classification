import streamlit as st
from PIL import Image

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
        The main goal is .....
        """
    , unsafe_allow_html=True)
    
    # Chemin d'accès à l'image
    image_path ="assets/full_process.jpg"

    # Lire l'image
    image_process = Image.open(image_path)

    # Afficher l'image
    st.image(image_process, caption='Full process', use_column_width=True)
