import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from passlib.hash import bcrypt
import requests
import asyncio


title = "User Login"
sidebar_name = "User Login"
dataPath = st.session_state.PrePath

# Chargement du fichier .env uniquement si le script est exécuté manuellement
if not os.getenv('GITHUB_ACTIONS'):
    dotenv_path = '../docker/.env'
    load_dotenv(dotenv_path)
if st.session_state.docker:
    dotenv_path = 'docker/.env'
    load_dotenv(dotenv_path)

# Récupération des tokens d'accès depuis les variables d'environnement
ACCESS_TOKEN_AUTH_0 = os.environ.get('ACCESS_TOKEN_AUTH_0')
ACCESS_TOKEN_AUTH_1 = os.environ.get('ACCESS_TOKEN_AUTH_1')
ACCESS_TOKEN_AUTH_2 = os.environ.get('ACCESS_TOKEN_AUTH_2')
MYSQL_HOST = st.session_state.users_db
MYSQL_USER = "root"
MYSQL_PASSWORD = os.getenv("MYSQL_ROOT_PWD")
MYSQL_DB = "rakuten_db"

def run():
    
    st.write()
    st.title(title)
    st.markdown('''
                ---
                ''')
    
    # Username input
    username = st.text_input("Username")

    # Password input (masked)
    password = st.text_input("Password", type="password")
    # Hash du mot de passe avec bcrypt
    hashed_password = bcrypt.hash(password)
     
    # Login button
    if st.button("Login"):
        # Vérification de username et password sur la base de données Utilisateurs MySQL
        try:
            connection = mysql.connector.connect(
                host=MYSQL_HOST,
                user=MYSQL_USER,
                port="3306", 
                password=MYSQL_PASSWORD,
                database=MYSQL_DB
            )
            if connection.is_connected():
                cursor = connection.cursor(dictionary=True)
                query = f"SELECT * FROM Users WHERE username = '{username}'"
                cursor.execute(query)
                user_connected = cursor.fetchone()
            return
        except Error as e:
            print(f"Error while querying MySQL: {e}")
        finally:
            cursor.close()
            connection.close()
            # if (user_connected["username"] == username) and (user_connected["password"]== hashed_password):
            if user_connected and bcrypt.verify(password, user_connected["password"]):
                st.success("Logged in successfully!")
                st.write()
                if (st.session_state.UserFirstName != user_connected["FirstName"]) and (st.session_state.UserLastName != user_connected["LastName"]):
                    st.session_state.username  = user_connected["username"]
                    st.session_state.UserFirstName = user_connected["FirstName"]
                    st.session_state.UserLastName = user_connected["LastName"]
                    st.session_state.UserAuthorization = user_connected["Authorization"]
                    # Generation de token
                    url = 'http://'+st.session_state.api_oauth + ':8001'
                    headers = {
                        'Content-Type': 'application/x-www-form-urlencoded',
                        }
                    data = {
                        'username': user_connected["username"],
                        'password': password,
                        }
                    response = requests.post(url+"/token", headers=headers, data=data)
                    st.session_state.token = response.json()["access_token"]
                    # st.experimental_rerun()
                st.write("Bienvenue "+user_connected["FirstName"]+" "+user_connected["LastName"]+" !")
                st.write("Vous êtes désormais connecté(e)")
                st.session_state.sale_step = 1
                
                st.write("")
                st.write("<strong>Base de Données Utilisateurs</strong>", unsafe_allow_html=True)
            # Récupérer tous les enregistrements de la base de données
                try:
                    connection = mysql.connector.connect(
                        host=MYSQL_HOST,
                        user=MYSQL_USER,
                        port="3306", 
                        password=MYSQL_PASSWORD,
                        database=MYSQL_DB
                    )
                    if connection.is_connected():
                        cursor = connection.cursor(dictionary=True)
                        query = "SELECT * FROM Users"
                        cursor.execute(query)
                        records = cursor.fetchall()
                        df = pd.DataFrame(records)
                        st.dataframe(df)
                except Error as e:
                    st.error(f"Error while querying MySQL: {e}")
                finally:
                    cursor.close()
                    connection.close()
            else:
                st.error("Invalid username or password. Please try again.")

    # Log off button
    if st.button("Log off"):
        st.write("Good bye "+st.session_state.username)
        st.write("Hope to see you soon !")
        st.session_state.username  = ""
        st.session_state.UserFirstName = ""
        st.session_state.UserLastName = ""
        st.session_state.UserAuthorization = 0
        st.session_state.token = ""
        st.success("Logged off successfully!")
