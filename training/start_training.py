import os
import sys

# Asegurar que el núcleo del modelo sea importable
sys.path.append(os.getcwd())

from CarleyCreativeCode.scripts.train_tokenizer import train_tokenizer
from CarleyCreativeCode.scripts.train_model import train

def start_workflow():
    print("--- INICIANDO FLUJO DE ENTRENAMIENTO CARLEY CREATIVE CODE ---")

    # 1. Entrenar Tokenizer si no existe
    if not os.path.exists("CarleyCreativeCode/tokenizer"):
        print("\n[1/2] Entrenando Tokenizer con tus datos en CarleyCreativeCode/data/...")
        train_tokenizer("CarleyCreativeCode/data", "CarleyCreativeCode/tokenizer")
    else:
        print("\n[1/2] Tokenizer ya existe. Saltando paso.")

    # 2. Iniciar Entrenamiento del Modelo
    print("\n[2/2] Iniciando entrenamiento del modelo (125M Parámetros)...")
    train()

    print("\n--- FLUJO COMPLETADO ---")
    print("Modelo guardado en: CarleyCreativeCode/carley_model.pt")

if __name__ == "__main__":
    start_workflow()
