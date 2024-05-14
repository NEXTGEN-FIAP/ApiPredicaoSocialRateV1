import numpy as np
from fastapi import FastAPI, Request
import pandas as pd
import joblib
import os

app = FastAPI()

@app.post("/modelo-predicao")
async def post(request: Request):
    data = await request.json()

    # Extrair os dados da solicitação JSON
    novos_dados = pd.DataFrame({
        'nicho': np.array([data.get("nicho")]),
        'publicoAlvoCampanha': np.array([data.get("publicoAlvoCampanha")]),
        'generoPublicoAlvoCampanha': np.array([data.get("generoPublicoAlvoCampanha")]),
        'interessesDoPublico': np.array([data.get("interessesDoPublico")]),
        'canalDaCampanha': np.array([data.get("canalDaCampanha")]),
        'alcanceDaCampanha': np.array([data.get("alcanceDaCampanha")])
    })

    modelo_path = "modelo_match_influencer.pkl"
    columns_path = "X_train_columns.pkl"

    if not os.path.exists(modelo_path) or not os.path.exists(columns_path):
        return {"message": "Modelo ou colunas de treinamento não encontradas"}

    try:
        # Carregar o modelo treinado
        modelo = joblib.load(modelo_path)
    except Exception as e:
        return {"message": f"Erro ao carregar o modelo: {str(e)}"}

    try:
        # Carregar as colunas usadas no treinamento do modelo
        X_train_columns = joblib.load(columns_path)
    except Exception as e:
        return {"message": f"Erro ao carregar as colunas de treinamento: {str(e)}"}

    # Converter variáveis categóricas em variáveis dummy
    novos_dados_dummificados = pd.get_dummies(novos_dados)

    # Garantir que as colunas nos novos dados sejam as mesmas que as usadas durante o treinamento
    novos_dados_dummificados = novos_dados_dummificados.reindex(columns=X_train_columns, fill_value=0)

    try:
        previsoes = modelo.predict(novos_dados_dummificados)
    except Exception as e:
        return {"message": f"Erro ao fazer a previsão: {str(e)}"}

    # Ensure previsoes is a simple value before returning
    if not isinstance(previsoes, (str, int, float, bool)):
        previsoes = str(previsoes)  # Convert to string if necessary

    return {"message": previsoes}