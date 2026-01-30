import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI(title="API Segittur")
templates = Jinja2Templates(directory="pages")

# 1. Definir el esquema
class HotelInput(BaseModel):
    AÑO: int
    MES: int
    PERIODO_ANTELACION_CODE: int
    CCAA_CODE: int
    PROVINCIA_CODE: int
    CATEGORIA_ALOJAMIENTO_CODE: int

# 2. CARGA DINÁMICA DEL MODELO
RUN_ID = "6c8fb1736a5f43d2a3d5a200214b36e0" 

# MLflow permite cargar el modelo usando esta URI especial
model_uri = f"runs:/{RUN_ID}/modelo_final"

try:
    # Usamos pyfunc para cargar cualquier sabor de modelo (sklearn, etc)
    model = mlflow.pyfunc.load_model(model_uri)
    model_status = "ok"
    print(f"✅ Modelo {RUN_ID} cargado correctamente desde MLflow")
except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
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