import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from passlib.hash import bcrypt
import requests
import asyncio
from extra_streamlit_components import tab_bar, TabBarItemData
import re
from passlib.hash import bcrypt


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

def display_user_db():
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
            st.link_button("Edit the MySQL database", "http://localhost:8080/?server=users_db&username=root&db=rakuten_db&select=Users")                
    except Error as e:
        st.error(f"Error while querying MySQL: {e}")
    finally:
        cursor.close()
        connection.close()
                
def run():
    
    st.write("")
    st.title(title)
    st.markdown('''
                ---
                ''')
    if (st.session_state.UserAuthorization>0):
        chosen_id = tab_bar(data=[
            TabBarItemData(id="tab1", title="Sign in", description=""),
            TabBarItemData(id="tab2", title="Join now", description=""),
            TabBarItemData(id="tab3", title="Cancel membership", description="")],
            default="tab1")
    else:
        chosen_id = tab_bar(data=[
            TabBarItemData(id="tab1", title="Sign in", description=""),
            TabBarItemData(id="tab2", title="Join now", description="")],
            default="tab1")
        
    if (chosen_id == "tab1"):
        # Username input
        username = st.text_input("Username", value=st.session_state.username)

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
                    password = ""
                    st.session_state.sale_step = 1
                    display_user_db()
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
        
        # If user logged in, display database    
        if st.session_state.username:
            display_user_db()

    if (chosen_id == "tab2"):
        pb=False
        password=""
        st.write("Enter your informations :")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        firstName = st.text_input("First Name")
        lastName = st.text_input("Last Name")
        email_regex = r'^[\w\-\.]+@([\w\-]+\.)+[\w\-]{2,4}$'
        email = st.text_input("email")
        if email:
            if re.match(email_regex, email) is None:
                pb = True
                st.error("Caution: email not valid")
        if st.button('Confirm your membership'):
            if username and password and firstName and lastName and email and not pb:
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
                        if username in df['username'].values:
                            st.error(f"Caution: username {username} already exists. User creation failed. Try again !")
                        else:
                            # Insert new user into the database
                            insert_query = "INSERT INTO Users (FirstName,LastName, Email, Authorization, username, password) VALUES (%s, %s, %s, %s, %s, %s)"
                            hashed_password = bcrypt.hash(password)
                            user_data = (firstName, lastName, email, 1, username, hashed_password )
                            cursor.execute(insert_query, user_data)
                            connection.commit()
            
                            st.success(f"User '{username}' successfully created!")
                            st.session_state.username  = username
                            st.session_state.UserFirstName = firstName
                            st.session_state.UserLastName = lastName
                            st.session_state.UserAuthorization = 1
                            # Generation de token
                            url = 'http://'+st.session_state.api_oauth + ':8001'
                            headers = {
                                'Content-Type': 'application/x-www-form-urlencoded',
                            }
                            data = {
                                'username': username,
                                'password': password,
                            }
                            response = requests.post(url+"/token", headers=headers, data=data)
                            st.session_state.token = response.json()["access_token"]
                            password = ""
                            display_user_db()
                except Error as e:
                    st.error(f"Error while querying MySQL: {e}")
                finally:
                    cursor.close()
                    connection.close()
            elif not (username and password and firstName and lastName and email):
                st.error("Information(s) missing")
 
    if (chosen_id == "tab3"):

        if (st.session_state.UserAuthorization>=1):
            cursor = None
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
                    if (st.session_state.UserAuthorization==1):
                        button_message = "Delete connected User"
                        username_to_delete = st.session_state.username
                    else:
                        button_message = "Delete User"
                            
                        # Retrieve the list of usernames from the database
                        cursor.execute("SELECT username FROM Users")
                        user_list = [row["username"] for row in cursor.fetchall()]
                        # Select username to delete
                        username_to_delete = st.selectbox("Select user to delete :",user_list)
                        
                    if st.button(button_message):    
                        # Check if the user exists in the database
                        query = "SELECT * FROM Users WHERE username = %s"
                        cursor.execute(query, (username_to_delete,))
                        user_to_delete = cursor.fetchone()
                            
                        if user_to_delete:
                            delete_query = "DELETE FROM Users WHERE username = %s"
                            cursor.execute(delete_query, (username_to_delete,))
                            connection.commit()
                            st.success(f"User '{username_to_delete}' has been successfully deleted.")
                            if (st.session_state.username==username_to_delete):
                                st.session_state.username = ""
                                st.session_state.UserFirstName = ""
                                st.session_state.UserLastName = ""
                                st.session_state.token = ""
                                st.session_state.UserAuthorization = 0
                        else:
                            st.error(f"User '{st.session_state.username}' does not exist.")
            except mysql.connector.Error as e:
                st.error(f"Error while querying MySQL: {e}")
            finally:
                if cursor:
                    cursor.close()
                if connection.is_connected():
                    connection.close()
            
                
            
