import os
import pandas as pd
from sklearn.model_selection import train_test_split

#ruta dataset default, relativa

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
"data",
"raw",
"WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def load_raw_data(csv_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Carga el dataset bruto de churn desde un archivo CSV.

    Parámetros
    ----------
    csv_path : str
        Ruta al archivo CSV con los datos originales.

    Devuelve
    --------
    pd.DataFrame
        DataFrame con los datos tal como vienen del archivo.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {csv_path}")

    df = pd.read_csv(csv_path)
    return df



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las transformaciones básicas de limpieza al DataFrame:

    - Convierte TotalCharges a numérico.
    - Elimina filas con valores faltantes en TotalCharges.
    - Elimina la columna customerID.
    - Mapea la columna Churn a 0/1.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.

    Devuelve
    --------
    pd.DataFrame
        DataFrame limpio y listo para separar en X e y.
    """
    data = df.copy()

    # Convertir TotalCharges a numérico, los errores los hago nulos
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Eliminar filas con NaN (son 11, pocas)
    data = data.dropna(subset=["TotalCharges"])

    # Eliminar ids
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # Mapear target a 0/1
    if "Churn" not in data.columns:
        raise ValueError("La columna 'Churn' no está presente en los datos.")

    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    return data


def split_features_target(data: pd.DataFrame):
    """
    Separa el DataFrame en matriz de características X y vector objetivo y.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame limpio que contiene la columna 'Churn'.

    Devuelve
    --------
    X : pd.DataFrame
        DataFrame con las variables categoricas.
    y : pd.Series
        Serie con la variable objetivo (0/1).
    """
    X = data.drop(columns=["Churn"])
    y = data["Churn"]
    return X, y


def get_train_test_data(
    csv_path: str = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Pipeline completo de preparación de datos para entrenamiento y test:

    1. Carga los datos brutos desde CSV.
    2. Limpia y transforma el DataFrame.
    3. Separa en X (features) e y (target).
    4. Divide en conjunto de entrenamiento y test, estratificando por y.

    Parámetros
    ----------
    csv_path : str
        Ruta al archivo CSV con los datos originales.
    test_size : float
        Proporción de datos reservados para el conjunto de test.
    random_state : int
        Semilla para reproducibilidad de la partición.

    Devuelve
    --------
    X_train, X_test, y_train, y_test
        Particiones listas para entrenar y evaluar el modelo.
    """
    df_raw = load_raw_data(csv_path)
    df_clean = clean_data(df_raw)
    X, y = split_features_target(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    # Bloque de prueba rápida
    X_train, X_test, y_train, y_test = get_train_test_data()
    print("Tamaños de las particiones:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)