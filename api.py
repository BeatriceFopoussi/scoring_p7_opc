from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import uvicorn
from shap import Explainer
import shap

app = FastAPI()

# Loading the model and data
model = pickle.load(open('models.pkl', 'rb'))
data = pd.read_csv('data_test.csv')
data_train = pd.read_csv('data_train.csv')
data_test_features = data.drop(columns=['TARGET'])

# Data processing
cols = data.select_dtypes(['float64']).columns
data_scaled = data.copy()
data_scaled[cols] = StandardScaler().fit_transform(data[cols])

cols = data_train.select_dtypes(['float64']).columns
data_train_scaled = data_train.copy()
data_train_scaled[cols] = StandardScaler().fit_transform(data_train[cols])
classifier = model.named_steps['classifier']
explainer = Explainer(classifier)

@app.get('/')
def welcome():
    return 'Welcome to the API'

@app.get('/{client_id}')
def check_client_id(client_id: int):
    if client_id in list(data['SK_ID_CURR']):
        return True
    else:
        return False

@app.get('/prediction/{client_id}')
def get_prediction(client_id: int):
    client_data = data[data['SK_ID_CURR'] == client_id]
    info_client = client_data.drop('SK_ID_CURR', axis=1)
    prediction = model.named_steps['classifier'].predict_proba(info_client)[0][1]
    return prediction

@app.get('/interpretabilite/{client_id}')
def get_shap_values(client_id: int):
    individual = data_test_features.iloc[[0]]
    shap_values_individual = explainer(individual)
    shap_values_individual_df = pd.DataFrame(shap_values_individual.values, columns=data_test_features.columns)
    shap_values_dict = shap_values_individual_df.to_dict(orient='records')[0]
    
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values_individual.values, individual)  

    return {"shap_values": shap_values_dict}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)
