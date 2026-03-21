import argparse
import torch
from CarleyCreativeCode.src.inference import CarleyInference

def main():
    parser = argparse.ArgumentParser(description="Carley Creative Code Inference Utility")
    parser.add_argument("--prompt", type=str, help="Código inicial para completar")
    parser.add_argument("--prefix", type=str, help="Código antes del error (para corrección)")
    parser.add_argument("--suffix", type=str, help="Código después del error (para corrección)")
    parser.add_argument("--model", type=str, default="CarleyCreativeCode/carley_model.pt")
    parser.add_argument("--tokens", type=str, default="CarleyCreativeCode/tokenizer")

    args = parser.parse_args()

    # CarleyInference now automatically loads config from config.json if it exists
    carley = CarleyInference(args.model, args.tokens)

    if args.prefix and args.suffix:
        print("--- MODO CORRECCIÓN (FIM) ---")
        result = carley.fix_code(args.prefix, args.suffix)
        print(f"Propuesta de corrección: {result}")
    elif args.prompt:
        print("--- MODO GENERACIÓN ---")
        result = carley.generate(args.prompt)
        print(f"Código generado: {result}")
    else:
        print("Por favor, proporciona un --prompt o --prefix y --suffix")

if __name__ == "__main__":
    main()
