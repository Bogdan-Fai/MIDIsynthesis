from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from clearml import Task, Dataset

from src.dataset import MIDIDataset
from src.model import MIDITransformer


def load_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


def train(task=None):
    if task is None:
        task = Task.current_task()

    config = {
        "block_size": 128,
        "batch_size": 32,
        "epochs": 10,
        "lr": 3e-4,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
        "dropout": 0.2,
        "num_workers": 4,
    }

    config = task.connect(config, name="Hyperparameters")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = task.get_logger()

    dataset_obj = Dataset.get(
        dataset_project="MIDIsynthesis",
        dataset_name="midi_dataset",
    )
    local_path = dataset_obj.get_local_copy()
    token_ids = np.load(f"{local_path}/token_ids.npy").tolist()
    stoi = load_vocab(f"{local_path}/vocab.json")

    dataset = MIDIDataset(token_ids, config["block_size"])

    print(f"Dataset size: {len(dataset)}")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training set size: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=(device == "cuda"),
    )

    model = MIDITransformer(
        vocab_size=len(stoi),
        d_model=config["d_model"],
        block_size=config["block_size"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        logger.report_text(f"Epoch {epoch + 1}/{config['epochs']}")

        model.train()
        train_loss_sum = 0.0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            if step % 50 == 0:
                current_loss = loss.item()
                print(f"step {step}/{len(train_loader)} | loss={current_loss:.4f}")
                logger.report_scalar(
                    title="Loss",
                    series="train",
                    value=current_loss,
                    iteration=epoch * len(train_loader) + step
                )

        avg_train_loss = train_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                _, loss = model(x, y)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}"
        )

        logger.report_scalar("Loss", "train_avg", avg_train_loss, epoch)
        logger.report_scalar("Loss", "val_avg", avg_val_loss, epoch)

        best_val_loss = min(best_val_loss, avg_val_loss)

    output_dir = Path("Data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "midi_transformer.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "vocab_size": len(stoi),
            "best_val_loss": best_val_loss,
        },
        model_path,
    )

    task.upload_artifact(
        name="Final_Model",
        artifact_object=str(model_path),
    )

    print("Training finished. Model saved.")
    print(f"Task ID: {task.id}")