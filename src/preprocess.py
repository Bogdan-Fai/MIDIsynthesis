# preprocess.py
import os
import numpy as np

def preprocess(midis_path):
    from Services.tokenizer import Tokenizer
    if not os.path.exists(midis_path):
        print(f"Path '{midis_path}' doesn't exist!")
        return

    tokenizer = Tokenizer()

    tokens_all = []
    
    tokenizer.load_vocab("Data/vocab.json")

    for midi_file in os.listdir(midis_path):
        if midi_file.endswith(".mid"):
            tokens = tokenizer.tokenize_file(os.path.join(midis_path, midi_file), "ids")
            tokens_all.extend(tokens)

    np.save("Data/token_ids.npy", tokens_all)