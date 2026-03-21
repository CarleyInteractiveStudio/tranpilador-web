import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

def train_tokenizer(data_path, output_dir):
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>",
        "<PRE>", "<MID>", "<SUF>",
        "si", "sino", "publico", "numero", "variable", "y", "o", "no",
        "alActualizar", "teclaPresionada", "teclaRecienPresionada", "estaTocandoTag",
        "fisica", "reproducir", "nuevo", "Vector2"
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.js', '.cc', '.txt'))]

    if not files:
        example_file = os.path.join(data_path, "example.cc")
        with open(example_file, "w") as f:
            f.write('ve motor; publico numero velocidad = 300; si (teclaPresionada("d")) { fisica.velocity.x = 1; }')
        files = [example_file]

    tokenizer.train(files, trainer)
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        clean_up_tokenization_spaces=True,
        bos_token="<BOS>",
        eos_token="<EOS>",
        unk_token="<UNK>",
        pad_token="<PAD>",
    )
    fast_tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer guardado en {output_dir}")

if __name__ == "__main__":
    train_tokenizer("CarleyCreativeCode/data", "CarleyCreativeCode/tokenizer")
