import json
import requests
import os
import sys

API_URL = "http://127.0.0.1:8000/predict_churn"

def load_customer_from_json(path: str) -> dict:
    """
    Lee un archivo JSON y devuelve el diccionario con los datos del cliente.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("El JSON debe contener un objeto.")

    return data

def call_churn_api(customer_dict: dict) -> dict:
    """
    Envía los datos de un cliente a la API de churn
    y devuelve la respuesta como diccionario de Python.

    Si la API responde con un código distinto de 200,
    lanza una excepción con información del error.
    """
    response = requests.post(API_URL, json=customer_dict)

    if response.status_code != 200:
        raise RuntimeError(
            f"Error al llamar a la API.\n"
            f"Status code: {response.status_code}\n"
            f"Respuesta: {response.text}"
        )

    return response.json()


def print_result(customer_dict: dict, result_dict: dict) -> None:
    """
    Imprime por pantalla un resumen legible de:
    - los datos del cliente
    - la probabilidad de churn predicha
    - la clasificación binaria y el nivel de riesgo
    """
    print("=== Datos del cliente ===")
    print(json.dumps(customer_dict, indent=4, ensure_ascii=False))

    print("\n=== Resultado del modelo ===")
    prob = result_dict["churn_probability"]
    pred = result_dict["churn_prediction"]
    risk = result_dict["risk_label"]

    print(f"Probabilidad de churn: {prob:.3f}")
    print(f"Predicción (0 = no churn, 1 = churn): {pred}")
    print(f"Nivel de riesgo: {risk}")


if __name__ == "__main__":
    #Construimos un cliente

    if len(sys.argv) >= 2:
        customer_path = sys.argv[1]  
    else:
        customer_path=input("Ingrese el path del Json del customer: ").strip()
    


    #Cargamos cliente
    try:
        customer = load_customer_from_json(customer_path)

        #Llamamos a la API de churn
        result = call_churn_api(customer)

        #Mostramos el resultado de forma legible
        print_result(customer, result)
    except Exception as e:
        print(f'\nOcurrio un error {e} (Recuerde path arranca desde src)')






