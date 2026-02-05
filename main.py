import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import io
import os
import joblib
from databricks.sdk import WorkspaceClient

app = FastAPI(title="API Segittur")
templates = Jinja2Templates(directory="pages")

class HotelInput(BaseModel):
    AÑO: int
    MES: int
    PERIODO_ANTELACION_CODE: int
    CCAA_CODE: int
    PROVINCIA_CODE: int 
    CATEGORIA_ALOJAMIENTO_CODE: int

DB_HOST = "https://dbc-1a9bf387-9846.cloud.databricks.com"
DB_TOKEN = os.getenv("DATABRICKS_TOKEN")
ruta_volumen = "/Volumes/workspace/default/modeliapro/model.pkl"

try:
    w = WorkspaceClient(host=DB_HOST, token=DB_TOKEN)

    print("Accediendo al modelo en la nube...")
    response = w.files.download(ruta_volumen)
    
    model_bytes = response.contents.read()
    
    if not model_bytes:
        raise ValueError("El archivo da error.")

    buffer = io.BytesIO(model_bytes)
    model = joblib.load(buffer)
    
    model_status = "ok"
    print("Modelo cargado...")

except Exception as e:
    print(f"Error: {e}")
    model_status = "ko"

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": model_status}

@app.post("/predict")
def predict(input_data: HotelInput):
    if model_status == "ko":
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    df_input = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df_input)
    
    return {
        "prediccion": int(prediction[0]),
        "categoria": "Caro" if prediction[0] == 1 else "Barato"
    }