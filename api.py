# Library imports
from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import uvicorn
from sklearn.preprocessing import LabelEncoder

# Create a FastAPI instance
app = FastAPI()

# Loading the model and data
model = pickle.load(open('models.pkl', 'rb'))  # Chargement du pipeline complet

data = pd.read_csv('data_test.csv')
data_train = pd.read_csv('data_train.csv')

# Traitement des données
cols = data.select_dtypes(['float64']).columns
data_scaled = data.copy()
data_scaled[cols] = StandardScaler().fit_transform(data[cols])

cols = data_train.select_dtypes(['float64']).columns
data_train_scaled = data_train.copy()
data_train_scaled[cols] = StandardScaler().fit_transform(data_train[cols])

# Initialisation de l'explainer SHAP pour le classificateur
explainer = shap.TreeExplainer(model.named_steps['classifier'])  # Utilisation du classificateur dans le pipeline

# Functions
@app.get('/')
def welcome():
    """
    Welcome message.
    :param: None
    :return: Message (string).
    """
    return 'Welcome to the API'


@app.get('/{client_id}')
def check_client_id(client_id: int):
    """
    Customer search in the database
    :param: client_id (int)
    :return: message (string).
    """
    if client_id in list(data['SK_ID_CURR']):
        return True
    else:
        return False


@app.get('/prediction/{client_id}')
def get_prediction(client_id: int):
    """
    Calculates the probability of default for a client.
    :param: client_id (int)
    :return: probability of default (float).
    """
    client_data = data[data['SK_ID_CURR'] == client_id]
    info_client = client_data.drop('SK_ID_CURR', axis=1)

    # Prévision avec le pipeline complet
    prediction = model.named_steps['classifier'].predict_proba(info_client)[0][1]
    
    return prediction



if __name__ == '__main__':
    # Démarrer le serveur FastAPI
    uvicorn.run(app, host="127.0.0.1", port=8001)
