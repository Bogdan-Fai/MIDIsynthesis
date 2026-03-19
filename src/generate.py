import json
import numpy as np
import torch
from model import MIDITransformer


def load_vocab(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab["stoi"]
    itos = {int(k): v for k, v in vocab["itos"].items()}
    return stoi, itos


def decode_tokens(token_ids, itos):
    return [itos[int(i)] for i in token_ids]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stoi, itos = load_vocab("Data/processed/vocab.json")

    model = MIDITransformer(
        vocab_size=len(stoi),
        d_model=256,
        block_size=128,
        n_heads=8,
        n_layers=6,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load("Data/processed/midi_transformer.pt", map_location=device))
    model.eval()

    start_token = stoi["START"]
    x = torch.tensor([[start_token]], dtype=torch.long, device=device)

    generated = model.generate(x, max_new_tokens=200)[0].tolist()
    tokens = decode_tokens(generated, itos)

    print(tokens)


if __name__ == "__main__":
    main()