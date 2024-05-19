import os
import requests
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# Chargement du fichier .env uniquement si le script est exécuté manuellement
if not os.getenv('GITHUB_ACTIONS'):
    dotenv_path = 'docker/.env'
    load_dotenv(dotenv_path)

# Récupération des tokens d'accès depuis les variables d'environnement
ACCESS_TOKEN_AUTH_0 = os.environ.get('ACCESS_TOKEN_AUTH_0')
ACCESS_TOKEN_AUTH_1 = os.environ.get('ACCESS_TOKEN_AUTH_1')
ACCESS_TOKEN_AUTH_2 = os.environ.get('ACCESS_TOKEN_AUTH_2')
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = os.getenv("MYSQL_ROOT_PWD")
MYSQL_DB = "rakuten_db"

def test_calc_addition():
    # Fonction test du résultat de 2+4
    output = 2 + 4
    assert output == 6
    
def test_MySQL():
    
    # Vérification du fonctionnement de la base de données Utilisateurs MySQL
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
            query = f"SELECT * FROM Users WHERE username = 'Alice'"
            cursor.execute(query)
            user_connected = cursor.fetchone()
            ###1.1  Vérification du fonctionnement de la base de données Utilisateurs MySQL avec username == 'Alice'###
            assert user_connected["username"] == 'Alice'
            ###1.2  Vérification du fonctionnement de la base de données Utilisateurs MySQL avec LastName == 'Sapritch'###
            assert user_connected["LastName"] == 'Sapritch'
            return
    except Error as e:
        print(f"Error while querying MySQL: {e}")
    finally:
        cursor.close()
        connection.close()
        

def test_access_authorization():
    
    url = 'http://localhost:8001'
    
    # 1 - Vérification du fonctionnement de l'API
    response = requests.get(url)
    ###2.1 - Vérification du fonctionnement de l'API acces sécurisé###
    assert response.status_code == 200
    
    # 2 - Vérification de la génération de token
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = {
        'username': 'Alice',
        'password': 'Alice',
    }
    response = requests.post(url+"/token", headers=headers, data=data)
    ###2.2.a - Vérification du processus de génération de token###
    assert response.status_code == 200
    ###2.2.b - Vérification de la valeur du token pour Alice###
    assert response.json()["access_token"]==ACCESS_TOKEN_AUTH_0
    
    # Vérification de l'accès sécurisé en utilisant le token d'accès
    # 3 - Authorization niveau 0 = Pas d'acces au Train du modèle ni à la prediction
    response = requests.get(url+"/secured", headers={"Authorization": f"Bearer {ACCESS_TOKEN_AUTH_0}"})
    ###2.3 - Authorisation niveau 0 pour Alice => Pas d'acces au Train du modèle ni à la predicti###
    assert response.status_code == 200
    
    # 4 - Authorization niveau 1 = Accès seulement à la prediction, mais pas d'acces au Train du modèle
    response = requests.get(url+"/secured", headers={"Authorization": f"Bearer {ACCESS_TOKEN_AUTH_1}"})
    ###2.4 - Authorisation niveau 1 pour John => Accès seulement à la prediction, mais pas d'acces au Train du modèle###
    assert response.status_code == 200
    
    # 5 - Authorization niveau 2 = Accès à la prediction et au Train du modèle
    response = requests.get(url+"/secured", headers={"Authorization": f"Bearer {ACCESS_TOKEN_AUTH_2}"})
    ###2.5 - Authorisation niveau 2 pour Fadimatou => Accès à la prediction et au Train du modèle###
    assert response.status_code == 200
    
def test_predict():
    
    url = 'http://localhost:8000'
    
    # 1 - Vérification de l'initialisation de l'API
    response = requests.get(url+"/initialisation")
    ###3.1 - Vérification de l'initialisation de l'API###
    assert response.status_code == 200 
    
    # 2 - Vérification de la predicition en mode unsecured
    headers={'Content-Type': 'application/json',
             "Authorization": f"Bearer {ACCESS_TOKEN_AUTH_1}"
             }
    data={"dataset_path": "tests/predict/X_test_update.csv",
          "images_path": "tests/predict/image_test",
          "prediction_path": "tests/predict",
          "api_secured": False,
          }
    response = requests.post(url+"/prediction", headers=headers, json=data)
    # Vérifier si la réponse est valide avant de la décoder en JSON
    if response.status_code == 200:
        try:
            response_json = response.json()
        except ValueError:
            print("Response is not in JSON format")
    else:
        print("Request failed with status code:", response.status_code)
        print("Response Text:", response.text)

    ###3.2 - Vérification de la predicition en mode unsecured###
    assert response.status_code == 200
    
    # 3 - Vérification de la predicition en mode secured avec un niveau d'autorisation = 1
    headers={'Content-Type': 'application/json',
             "Authorization": f"Bearer {ACCESS_TOKEN_AUTH_1}"
             }
    data["api_secured"] = True
    response = requests.post(url+"/prediction", headers=headers, json=data)
    ###3.3 - Vérification de la predicition en mode secured avec un niveau d'autorisation = 1###
    assert (response.status_code == 200)
    
    # 4 - Vérification du refus de predicition en mode secured avec un niveau d'autorisation = 0
    headers={'Content-Type': 'application/json',
             "Authorization": f"Bearer {ACCESS_TOKEN_AUTH_0}"
             }
    response = requests.post(url+"/prediction", headers=headers, json=data)
    ###3.4 - Vérification du refus de predicition en mode secured avec un niveau d'autorisation = 0###
    assert response.status_code == 403
    
