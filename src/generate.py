import json
import numpy as np
import torch
from src.model import MIDITransformer

def load_vocab(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab
    itos = {int(v): k for k, v in vocab.items()}
    return stoi, itos


def decode_tokens(token_ids, itos):
    return [itos[int(i)] for i in token_ids]


def generate(task=None):
    from Services.midi_service import save_midi_from_stream, play_midi_from_stream
    from Services.miditxt_converter import make_midi_stream

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = task.get_logger() if task is not None else None

    stoi, itos = load_vocab("Data/vocab.json")

    checkpoint = torch.load("Data/outputs/midi_transformer_final.pt", map_location=device)
    saved_config = checkpoint["config"]

    model = MIDITransformer(
        vocab_size=checkpoint["vocab_size"],
        d_model=saved_config["d_model"],
        block_size=saved_config["block_size"],
        n_heads=saved_config["n_heads"],
        n_layers=saved_config["n_layers"],
        dropout=saved_config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    generate_parameters = {
        "start_token": [stoi["START"], stoi["A2_C#3_E3_G3_2.0"]],
        "end_token_id": stoi["END"],
        "max_new_tokens": 256,
        "temperature": 0.9,
        "top_k": 16
    }

    generate_parameters = task.connect(generate_parameters, name="Generation Parameters") if task else generate_parameters

    context = torch.tensor([generate_parameters["start_token"]], dtype=torch.long, device=device)

    generated = model.generate(
        idx=context,
        end_token_id=generate_parameters["end_token_id"],
        max_new_tokens=generate_parameters["max_new_tokens"],
        temperature=generate_parameters["temperature"],
        top_k=generate_parameters["top_k"],
    )[0].tolist()
    tokens = decode_tokens(generated, itos)

    print(tokens)

    midi_path = save_midi_from_stream(make_midi_stream(tokens), path="Data/outputs")

    if logger:
        logger.report_text("Generation completed, saved to: " + str(midi_path))
    if task:
        task.upload_artifact(name="data", artifact_object=midi_path)