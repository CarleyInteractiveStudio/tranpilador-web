import os
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from CarleyCreativeCode.src.model import CarleyCreativeCodeModel, Config
from CarleyCreativeCode.src.dataset import FIMDataset
from accelerate import Accelerator

def train():
    accelerator = Accelerator()
    device = accelerator.device

    vocab_size = 32000
    n_embd = 768
    n_layer = 12
    n_head = 12
    block_size = 1024
    batch_size = 8
    learning_rate = 5e-4
    epochs = 1

    tokenizer = PreTrainedTokenizerFast.from_pretrained("CarleyCreativeCode/tokenizer")

    config = Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, block_size=block_size)
    model = CarleyCreativeCodeModel(config)

    dataset = FIMDataset("CarleyCreativeCode/data", tokenizer, block_size=block_size)
    print(f"Dataset length: {len(dataset)}")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits, loss = model(x, y)
            accelerator.backward(loss)
            optimizer.step()

            if step % 1 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "CarleyCreativeCode/carley_model.pt")
        # También guardar configuración para que transformers lo reconozca
        import json
        config_dict = {
            "vocab_size": config.vocab_size,
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "block_size": config.block_size,
            "model_type": "gpt2"
        }
        with open("CarleyCreativeCode/config.json", "w") as f:
            json.dump(config_dict, f)

        print("Modelo y configuración guardados correctamente.")

if __name__ == "__main__":
    train()
