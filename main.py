import requests
from fastapi import FastAPI, Request, Response
from joblib import load


app = FastAPI()

modelo_predicao = load("modelo_match_influencer.pkl")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/modelo-predicao")
async def post(request: Request):
    data = await request.json()
    nicho = data.get("nicho")
    publico_alvo = data.get("publicoAlvoCampanha")
    genero_alvo = data.get("generoAlvoCampanha")
    interesses = data.get("interessesDoPublico")
    canal_campanha = data.get("canalDaCampanha")
    alcance_campanha = data.get("alcanceDaCampanha")
    return {"message": data["nicho"]}
