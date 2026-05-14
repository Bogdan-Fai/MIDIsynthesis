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
        return json.load(f)


def save_checkpoint(task, model, optimizer, epoch, config, vocab_size, best_val_loss):
    output_dir = Path("Data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
            "vocab_size": vocab_size,
            "best_val_loss": best_val_loss,
        },
        ckpt_path,
    )

    task.upload_artifact(
        name=f"checkpoint_epoch_{epoch + 1}",
        artifact_object=str(ckpt_path),
    )


def load_checkpoint(resume_task_id: str, resume_artifact_name: str, device: str):
    prev_task = Task.get_task(task_id=resume_task_id)
    ckpt_path = prev_task.artifacts[resume_artifact_name].get_local_copy()
    checkpoint = torch.load(ckpt_path, map_location=device)
    return checkpoint


def train(task=None, resume_task_id=None, resume_artifact_name=None):
    if task is None:
        task = Task.current_task()

    config = {
        "block_size": 128,
        "batch_size": 32,
        "epochs": 15,
        "lr": 3e-4,
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
        "dropout": 0.2,
        "num_workers": 0,
        "weight_decay": 1e-2,
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
    split_idx = int(0.9 * len(token_ids))

    train_dataset = token_ids[:split_idx]
    val_dataset = token_ids[split_idx:]

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_task_id and resume_artifact_name:
        checkpoint = load_checkpoint(
            resume_task_id=resume_task_id,
            resume_artifact_name=resume_artifact_name,
            device=device,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"Resume from epoch {start_epoch}")

    for epoch in range(start_epoch, config["epochs"]):
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

            if step % 20 == 0:
                iteration = epoch * len(train_loader) + step
                logger.report_scalar("Loss", "train", loss.item(), iteration)
                logger.report_text(f"alive | epoch={epoch + 1} step={step}")

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
        best_val_loss = min(best_val_loss, avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}"
        )

        logger.report_scalar("Loss", "train_avg", avg_train_loss, epoch)
        logger.report_scalar("Loss", "val_avg", avg_val_loss, epoch)

        save_checkpoint(
            task=task,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            vocab_size=len(stoi),
            best_val_loss=best_val_loss,
        )

    output_dir = Path("Data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = output_dir / "midi_transformer_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "vocab_size": len(stoi),
            "best_val_loss": best_val_loss,
        },
        final_model_path,
    )

    task.upload_artifact("Final_Model", artifact_object=str(final_model_path))

    print("Training finished. Model saved.")
    print(f"Task ID: {task.id}")


if __name__ == "__main__":
    train()