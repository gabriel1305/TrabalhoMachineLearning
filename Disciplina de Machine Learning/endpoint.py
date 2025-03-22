import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

os.chdir(os.path.abspath(os.curdir))

model=joblib.load('model_tree.pkl')
app = FastAPI(title="API de precisão de Preço de imóvel")


model_columns = [ 'Valor do Imóvel','Quantidade de Cômodos', 'Vagas de Garagem',
       'Área (m²)', 'Ano de Construção', 'Valor do Condomínio',
       'Distância do Centro (km)', 'Idade_Imovel(Anos)']

class ImputData(BaseModel):
    features: list[float]

@app.post('/predict')
def predict(data: ImputData):
    input_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_array)
    return {'Valor_previsto': round(float(prediction[0]), 2)}

