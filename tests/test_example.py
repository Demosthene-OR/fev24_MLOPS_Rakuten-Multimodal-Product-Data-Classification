import os
import pytest
import requests
# import env_config
from dotenv import load_dotenv
import os

dotenv_path = 'docker/.env'
load_dotenv(dotenv_path)

# Maintenant, vous pouvez accéder aux variables d'environnement normalement
ACCESS_TOKEN_AUTH_0 = os.environ.get('ACCESS_TOKEN_AUTH_0')
# Récupération des tokens d'accès depuis les variables d'environnement
ACCESS_TOKEN_AUTH_0 = os.environ.get('ACCESS_TOKEN_AUTH_0')
ACCESS_TOKEN_AUTH_1 = os.environ.get('ACCESS_TOKEN_AUTH_1')
ACCESS_TOKEN_AUTH_2 = os.environ.get('ACCESS_TOKEN_AUTH_2')

def test_calc_addition():
    # Fonction test du résultat de 2+4
    output = 2 + 4
    assert output == 6
    
def test_access_authorization_0():
    # Vérification de l'accès sécurisé en utilisant le token d'accès
    # Authorization niveau 0 = Pas d'acces au Train du modèle ni à la prediction
    auth_response = requests.get("http://localhost:8001/secured", headers={"Authorization": f"Bearer {ACCESS_TOKEN_AUTH_0}"})
    assert auth_response.status_code == 200

def test_access_authorization_1():
    # Authorization niveau 1 = Accès seulement à la prediction, mais pas d'acces au Train du modèle
    auth_response = requests.get("http://localhost:8001/secured", headers={"Authorization": f"Bearer {ACCESS_TOKEN_AUTH_1}"})
    assert auth_response.status_code == 200

def test_access_authorization_2():
    # Authorization niveau 1 = Accès à la prediction et au Train du modèle
    auth_response = requests.get("http://localhost:8001/secured", headers={"Authorization": f"Bearer {ACCESS_TOKEN_AUTH_2}"})
    assert auth_response.status_code == 200
    

