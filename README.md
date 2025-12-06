# customer-churn-ml-project

# Customer Churn Prediction (Telco)

Proyecto end-to-end de Data Science para predecir la probabilidad de abandono (churn) de clientes de una empresa de telecomunicaciones y exponer el modelo mediante una API REST.

El objetivo es que, dado el perfil de un cliente, el sistema devuelva:

- probabilidad de churn
- predicción binaria (0 = no churn, 1 = churn)
- nivel de riesgo (`bajo`, `medio`, `alto`) (evaluado relativamente)

---

## Tecnologías

- Python (pandas, numpy, scikit-learn)
- FastAPI + Uvicorn
- joblib (serialización del modelo)

---

## Modelo

- Problema: clasificación binaria (`Churn: Yes/No`).
- Modelo principal: **Regresión Logística** con pipeline de preprocesamiento:
  - numéricas escaladas con `StandardScaler`
  - categóricas con `OneHotEncoder`
- Métricas (test):
  - AUC ≈ **0.835**
  - Recall clase churn ≈ **0.80**

Se probó también un modelo de árboles (Random Forest), pero se mantuvo la regresión logística por mejor resultados al momento del recall del churn 1.

---

## Estructura

```text
customer-churn-ml-project/
├─ data/
│  └─ clients_examples/         # Ejemplos de clientes en JSON
├─ models/
│  └─ churn_model.joblib        # Modelo entrenado (pipeline completo)
├─ notebooks/
│  └─ churn_eda_and_modeling.ipynb
├─ src/
│  ├─ data_preparation.py       # Carga y train/test split
│  ├─ train_model.py            # Entrenamiento y guardado del modelo
│  ├─ client_example.py         # Cliente CLI para consumir la API
│  └─ api/
│     └─ app.py                 # API FastAPI (/predict_churn)
└─ README.md
```
## Cómo ejecutar

### 1. Crear y activar el entorno virtual

Desde la carpeta raíz del proyecto:

    python -m venv venv
    source venv/Scripts/activate    # en Windows + Git Bash

Instalar dependencias:

    pip install -r requirements.txt


### 2. Entrenar el modelo (opcional)

Si quiere volver a entrenar el modelo y regenerar el archivo .joblib:

    python -m src.train_model

Este comando:
- carga y prepara los datos (usando src/data_preparation.py),
- entrena el pipeline de regresión logística,
- calcula métricas en el set de test,
- guarda el modelo entrenado (preprocesamiento + clasificador) en:

    models/churn_model.joblib

La serialización se hace con joblib.dump(...) dentro de src/train_model.py.


### 3. Levantar la API (FastAPI + Uvicorn)

Con el entorno virtual activo, en la raíz del proyecto:

    python -m uvicorn src.api.app:app --reload

La API queda disponible en:

- Root:  http://127.0.0.1:8000/
- Docs:  http://127.0.0.1:8000/docs

En src/api/app.py la función load_model() usa joblib.load(...) para cargar models/churn_model.joblib una sola vez y reutilizarlo en cada request.


### 4. Consumir la API desde el cliente en Python (cliente CLI con JSON)

Ejemplo de archivo de entrada (data/clients_examples/client_1.json):

    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 5,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "No",
      "DeviceProtection": "Yes",
      "TechSupport": "No",
      "StreamingTV": "Yes",
      "StreamingMovies": "Yes",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 85.5,
      "TotalCharges": 300.75
    }

Con la API levantada en otra terminal, ejecutar el cliente:

    source venv/Scripts/activate
    python -m src.client_example data/clients_examples/client_1.json

Salida ejemplo:

    === Datos del cliente enviados a la API ===
    { ... }

    === Resultado del modelo ===
    Probabilidad de churn: 0.904
    Predicción (0 = no churn, 1 = churn): 1
    Nivel de riesgo: alto


### 5. Por realizar

- Integrar el modelo con una base de datos SQL (por ejemplo PostgreSQL) para leer y guardar clientes.
- Crear un pequeño dashboard (por ejemplo con Streamlit) para que usuarios de negocio puedan cargar un lote de clientes y ver su riesgo de churn.
- Ajustar umbrales de decisión y calibrar probabilidades según los costos de negocio (costo de perder un cliente vs costo de intervenir).
