import torch
import os
from transformers import PreTrainedTokenizerFast
from CarleyCreativeCode.src.model import CarleyCreativeCodeModel, Config

class CarleyInference:
    def __init__(self, model_path, tokenizer_path, config_params=None):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # Load from config.json if available, otherwise use defaults
        config_file = os.path.join(os.path.dirname(model_path), "config.json")
        if os.path.exists(config_file):
            import json
            with open(config_file, "r") as f:
                c_dict = json.load(f)
                # Remove extra keys that Config doesn't accept
                valid_keys = {"vocab_size", "n_embd", "n_layer", "n_head", "block_size", "dropout"}
                c_dict = {k: v for k, v in c_dict.items() if k in valid_keys}
                config = Config(**c_dict)
        elif config_params is not None:
            config = Config(**config_params)
        else:
            # Default 125M config
            config = Config(
                vocab_size=32000,
                n_embd=768,
                n_layer=12,
                n_head=12,
                block_size=1024
            )

        self.model = CarleyCreativeCodeModel(config)

        # Load state dict if exists
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path))
            self.model.to("cuda")
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        self.model.eval()

    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_k=40):
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens, temperature, top_k)

        return self.tokenizer.decode(output_ids[0])

    def fix_code(self, prefix, suffix, max_new_tokens=50):
        # Fill-In-The-Middle mode
        # <PRE> prefix <SUF> suffix <MID>
        prompt = f"<PRE>{prefix}<SUF>{suffix}<MID>"
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, max_new_tokens)

        full_text = self.tokenizer.decode(output_ids[0])
        # Extract only the middle part
        if "<MID>" in full_text:
            middle_part = full_text.split("<MID>")[-1]
            if "<EOS>" in middle_part:
                middle_part = middle_part.split("<EOS>")[0]
            return middle_part
        return full_text
