# tests/conftest.py
import pytest
from dotenv import load_dotenv
import os

@pytest.fixture(autouse=True)
def load_env():
    # Définir le chemin absolu vers le fichier .env dans le dossier docker
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'docker', '.env')
    load_dotenv(dotenv_path)
