import torch
from torch.utils.data import Dataset
import random
import os

class FIMDataset(Dataset):
    def __init__(self, data_dir, tokenizer, block_size=1024, fim_rate=0.5):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.fim_rate = fim_rate

        # Cargar todos los archivos de texto/código
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.js', '.cc', '.txt'))]
        self.data = []
        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                self.data.append(f.read())

        # Concatenar todo y tokenizar (para simplificar en esta fase inicial)
        full_text = "\n<EOS>\n".join(self.data)
        self.tokens = tokenizer.encode(full_text)

        # Obtener IDs de tokens especiales
        self.fim_pre = tokenizer.convert_tokens_to_ids("<PRE>")
        self.fim_mid = tokenizer.convert_tokens_to_ids("<MID>")
        self.fim_suf = tokenizer.convert_tokens_to_ids("<SUF>")

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        chunk = self.tokens[start:end]

        # Aplicar Fill-In-The-Middle aleatoriamente
        if random.random() < self.fim_rate:
            return self.apply_fim(chunk)

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def apply_fim(self, chunk):
        if len(chunk) < 8: # Muy corto para FIM
            return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

        # Dividir el chunk en 3 partes: Prefijo, Medio, Sufijo
        p1 = random.randint(1, len(chunk) // 3)
        p2 = random.randint(p1 + 1, (2 * len(chunk)) // 3)

        prefix = chunk[:p1]
        middle = chunk[p1:p2]
        suffix = chunk[p2:]

        # Formato FIM: <PRE> Prefix <SUF> Suffix <MID> Middle
        new_chunk = [self.fim_pre] + prefix + [self.fim_suf] + suffix + [self.fim_mid] + middle

        # Ajustar al block_size
        new_chunk = new_chunk[:self.block_size]
        if len(new_chunk) < self.block_size:
            new_chunk += [self.tokenizer.pad_token_id] * (self.block_size - len(new_chunk))

        x = torch.tensor(new_chunk[:-1], dtype=torch.long)
        y = torch.tensor(new_chunk[1:], dtype=torch.long)
        return x, y
