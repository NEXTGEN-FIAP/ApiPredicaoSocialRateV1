import requests
from fastapi import FastAPI
from joblib import load


app = FastAPI()

modelo_predicao = load("modelo_match_influencer.pkl")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/modelo-predicao")
async def post():
    dados = await requests.json()
    return {"message": "Hello Post"}