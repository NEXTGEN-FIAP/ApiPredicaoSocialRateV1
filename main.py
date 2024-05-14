import numpy as np
from fastapi import FastAPI, Request
import pandas as pd
import joblib

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

    # Carregar o modelo treinado
    modelo = joblib.load("modelo_match_influencer.pkl")

    # Carregar as colunas usadas no treinamento do modelo
    X_train_columns = joblib.load("X_train_columns.pkl")

    # Converter variáveis categóricas em variáveis dummy
    novos_dados_dummificados = pd.get_dummies(novos_dados)

    # Garantir que as colunas nos novos dados sejam as mesmas que as usadas durante o treinamento
    novos_dados_dummificados = novos_dados_dummificados.reindex(columns=X_train_columns, fill_value=0)

    previsoes = modelo.predict(novos_dados_dummificados)

    # Ensure previsoes is a simple value before returning
    if not isinstance(previsoes, (str, int, float, bool)):
        previsoes = str(previsoes)  # Convert to string if necessary

    return {"message": previsoes}
