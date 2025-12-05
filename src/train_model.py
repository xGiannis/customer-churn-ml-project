import os

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_preparation import get_train_test_data


def build_preprocessing_and_model(X_train):
    """
    Construye el pipeline completo: preprocesamiento + modelo.

    - Estandariza variables numéricas.
    - One-hot encoding para variables categóricas.
    - Clasificador: Regresión Logística con class_weight='balanced' (por el desbalance que tiene Churn).
    """

    num_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X_train.select_dtypes(include=["object"]).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1,      #poder de procesamiento, utiliza cpu interna. ando bien de poder yo je
                ),
            ),
        ]
    )

    return model


def train_and_evaluate():
    """
    Entrena el modelo de churn y muestra métricas de evaluación sobre el set de test.
    Además, guarda el modelo entrenado en la carpeta 'models/'.
    """

    # 1. Obtener los datos ya limpios y particionados
    X_train, X_test, y_train, y_test = get_train_test_data()

    # 2. Construir el pipeline
    model = build_preprocessing_and_model(X_train)

    # 3. Entrenar
    print("Entrenando el modelo...")
    model.fit(X_train, y_train)

    # 4. Evaluar
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== Classification report (classe 1 = Churn) ===")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {auc:.3f}")

    # 5. Guardar el modelo entrenado
    project_root = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "churn_model.joblib")
    joblib.dump(model, model_path)

    print(f"\nModelo guardado en: {model_path}")

    return model, auc


if __name__ == "__main__":
    train_and_evaluate()
