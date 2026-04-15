import json
import numpy as np
import torch
from src.model import MIDITransformer
from Services.midi_service import save_midi_from_stream


def load_vocab(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab
    itos = {int(v): k for k, v in vocab.items()}
    return stoi, itos


def decode_tokens(token_ids, itos):
    return [itos[int(i)] for i in token_ids]


def generate(task=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = task.get_logger() if task is not None else None

    stoi, itos = load_vocab("Data/vocab.json")

    model = MIDITransformer(
        vocab_size=len(stoi),
        d_model=128,
        block_size=64,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(torch.load("Data/outputs/midi_transformer.pt", map_location=device))
    model.eval()

    start_token = stoi["START"]
    end_token = stoi["END"]
    x = torch.tensor([[start_token]], dtype=torch.long, device=device)

    generated = model.generate(
        x,
        end_token_id=end_token,
        max_new_tokens=200
    )[0].tolist()
    tokens = decode_tokens(generated, itos)

    print(tokens)

    print("Play generated MIDI?")

    if input("y/n: ") == "y":
        from Services.miditxt_converter import make_midi_stream
        from Services.midi_service import play_midi_from_stream

        s = make_midi_stream(tokens)
        play_midi_from_stream(s)

    save_midi_from_stream(make_midi_stream(tokens), path="Data/outputs/generated.mid")

    if logger:
        logger.report_text("Generation completed")
    if task:
        task.upload_artifact(name="data", artifact_object="Data/outputs/generated.mid")