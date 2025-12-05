import os
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# -----------------------------
# 1. Definición del esquema de entrada
# -----------------------------

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# -----------------------------
# 2. Carga del modelo entrenado
# -----------------------------

@lru_cache()
def load_model():
    """
    Carga el modelo entrenado (pipeline completo) desde la carpeta models.

    Uso de lru_cache para que se cargue una sola vez por proceso.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(project_root, "models", "churn_model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")

    model = joblib.load(model_path)
    return model


# -----------------------------
# 3. Inicialización de la app FastAPI
# -----------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API para predecir la probabilidad de churn de clientes de telecomunicaciones.",
    version="1.0.0",
)


# -----------------------------
# 4. Endpoints
# -----------------------------

@app.get("/")
def read_root():
    return {"message": "API de churn operativa. Use POST /predict_churn para obtener predicciones."}


@app.post("/predict_churn")
def predict_churn(customer: CustomerFeatures):
    """
    Recibe las características de un cliente y devuelve:

    - probabilidad de churn (entre 0 y 1)
    - predicción binaria (0 = no churn, 1 = churn)
    - etiqueta de riesgo cualitativa
    """
    model = load_model()

    # Convertir el objeto Pydantic a DataFrame de una fila
    customer_df = pd.DataFrame([customer.dict()])

    # Probabilidades y predicción
    proba = model.predict_proba(customer_df)[0][1]
    pred = int(model.predict(customer_df)[0])

    # Etiqueta de riesgo simple
    if proba >= 0.8:
        risk_label = "alto"
    elif proba >= 0.5:
        risk_label = "medio"
    else:
        risk_label = "bajo"

    return {
        "churn_probability": float(proba),
        "churn_prediction": pred,
        "risk_label": risk_label,
    }
