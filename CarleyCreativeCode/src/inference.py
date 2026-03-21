import torch
from transformers import PreTrainedTokenizerFast
from CarleyCreativeCode.src.model import CarleyCreativeCodeModel, Config

class CarleyInference:
    def __init__(self, model_path, tokenizer_path, config_params=None):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # Default 125M config if not provided
        if config_params is None:
            config = Config(
                vocab_size=32000,
                n_embd=768,
                n_layer=12,
                n_head=12,
                block_size=1024
            )
        else:
            config = Config(**config_params)

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
        try:
            middle_part = full_text.split("<MID>")[1].split("<EOS>")[0]
            return middle_part
        except IndexError:
            return full_text
