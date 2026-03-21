import argparse
import os
from huggingface_hub import HfApi, Repository
from transformers import PreTrainedTokenizerFast
import torch

def push_to_hub(repo_id, model_path, tokenizer_path):
    api = HfApi()

    print(f"Subiendo modelo a Hugging Face Hub: {repo_id}")

    # Crear el repositorio si no existe
    api.create_repo(repo_id=repo_id, exist_ok=True)

    # Subir el modelo guardado (.pt)
    if os.path.exists(model_path):
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="pytorch_model.bin",
            repo_id=repo_id,
        )
        print("Modelo (.pt) subido correctamente.")

    # Subir el Tokenizer completo
    if os.path.isdir(tokenizer_path):
        api.upload_folder(
            folder_path=tokenizer_path,
            repo_id=repo_id,
        )
        print("Tokenizer subido correctamente.")

    print(f"Todo listo! Puedes ver tu modelo en: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Carley Creative Code to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Ejemplo: tu_usuario/carley-creative-code")
    parser.add_argument("--model", type=str, default="CarleyCreativeCode/carley_model.pt")
    parser.add_argument("--tokenizer", type=str, default="CarleyCreativeCode/tokenizer")

    args = parser.parse_args()

    push_to_hub(args.repo_id, args.model, args.tokenizer)
