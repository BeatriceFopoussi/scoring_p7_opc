from fastapi.testclient import TestClient
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from fastapi import status
import json
from api import app

client = TestClient(app)
API_URL = "http://127.0.0.1:8001/"  # Définir l'URL de l'API si nécessaire

def test_read_main():
    """Test l'endpoint racine de l'API."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the API for credit scoring"}  # Adapter selon la vraie réponse

def test_check_client_id():
    """Test la fonction check_client_id() de l'API avec un client existant."""
    url = "/check_client_id/396899"
    response = client.get(url)
    assert response.status_code == 200
    assert response.json() is True  # Pas besoin de `json.loads()`, `response.json()` suffit

def test_check_client_id_2():
    """Test la fonction check_client_id() de l'API avec un client inexistant."""
    url = "/check_client_id/396899"
    response = client.get(url)
    assert response.status_code == 200
    assert response.json() is False

def test_get_prediction():
    """Test la fonction get_prediction() de l'API."""
    url = "/prediction/396899"
    response = client.get(url)
    assert response.status_code == 200
    assert isinstance(response.json(), float)  # Vérifier que c'est bien un float
    assert 0 <= response.json() <= 1  # La probabilité doit être entre 0 et 1

import unittest

class TestExample(unittest.TestCase):
    def test_addition(self):
        """Test unitaire basique."""
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()
