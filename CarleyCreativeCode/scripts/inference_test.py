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

    # Matching train_model config for testing
    config_params = {
        "vocab_size": 32000,
        "n_embd": 128,
        "n_layer": 4,
        "n_head": 4,
        "block_size": 16
    }

    carley = CarleyInference(args.model, args.tokens, config_params)

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
