import json
import numpy as np
import torch
from src.model import MIDITransformer
from clearml import Task

def load_vocab(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab
    itos = {int(v): k for k, v in vocab.items()}
    return stoi, itos


def decode_tokens(token_ids, itos):
    decoded = []
    for i in token_ids:
        token_id = int(i)
        if token_id in itos:
            decoded.append(itos[token_id])
        else:
            # Generate meaningful token names for unknown IDs
            # This creates tokens like "C4_0.5", "D4_1.0", etc. for unknown IDs
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            octaves = ['2', '3', '4', '5', '6']
            durations = ['0.25', '0.5', '0.75', '1.0', '1.5', '2.0', '3.0', '4.0']

            # Create a deterministic but varied token name based on ID
            note_idx = (token_id * 7) % len(notes)
            octave_idx = (token_id * 3) % len(octaves)
            duration_idx = (token_id * 5) % len(durations)

            note = notes[note_idx]
            octave = octaves[octave_idx]
            duration = durations[duration_idx]

            decoded.append(f"{note}{octave}_{duration}")
    return decoded


def generate(path="Data/outputs/midi_transformer_final_1.pt", task=None, seed=None):
    from Services.midi_service import save_midi_from_stream, play_midi_from_stream
    from Services.miditxt_converter import make_midi_stream
    import random
    import time

    stoi, itos = load_vocab("Data/vocab.json")


    seed = int(time.time() * 1_000_000) % (2**32)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    candidate_tokens = [
        token for token in stoi.keys()
        if token not in ["START", "END"]
    ]

    random_start_token = random.choice(candidate_tokens)

    generate_parameters = {
        "start_token": [stoi["START"], stoi[random_start_token]],
        "end_token_id": stoi["END"],
        "max_new_tokens": 256,
        "temperature": 1.8,
        "top_k": 100,
    }
    print("Prompt token:", random_start_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # logger = task.get_logger() if task is not None else None

    prev_task = Task.get_task(task_id="bfd1f69da2804b99931848ab93d852eb")
    model_path = prev_task.artifacts["Final_Model"].get_local_copy()
    checkpoint = torch.load(model_path, map_location=device)

    # Set random seed if provided
    # if seed is not None:
    #     import random
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if device == "cuda":
    #         torch.cuda.manual_seed_all(seed)

    # checkpoint = torch.load(path , map_location=device)
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

    # generate_parameters = task.connect(generate_parameters, name="Generation Parameters") if task else generate_parameters

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

    # if logger:
    #     logger.report_text("Generation completed, saved to: " + str(midi_path))
    if task:
        task.upload_artifact(name="data", artifact_object=midi_path)