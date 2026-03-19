import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import MIDIDataset
from src.model import MIDITransformer


def load_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi = vocab
    itos = {int(v): k for k, v in vocab.items()}
    return stoi, itos


def train():
    print("Starting training...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    token_ids = np.load("Data/token_ids.npy").tolist()
    stoi, itos = load_vocab("Data/vocab.json")

    block_size = 128
    batch_size = 32
    epochs = 10
    lr = 3e-4

    dataset = MIDIDataset(token_ids, block_size)

    print(f"Dataset size: {len(dataset)}")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("Initializing model...")
    model = MIDITransformer(
        vocab_size=len(stoi),
        d_model=256,
        block_size=block_size,
        n_heads=8,
        n_layers=4,
        dropout=0.2,
    ).to(device)

    print("Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                _, loss = model(x, y)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f}"
        )

    torch.save(model.state_dict(), "Data/midi_transformer.pt")
    print("Training finished. Model saved to Data/midi_transformer.pt")