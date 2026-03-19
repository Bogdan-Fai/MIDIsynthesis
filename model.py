# !pip install torch --quiet  # для Colab

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# предсказать следующий токен
# -----------------------------
class NextTokenDataset(Dataset):
    """
    Генерируем последовательности по простому правилу:
    start ~ U[0..V-1]
    seq[i] = (start + step*i) % V
    target = следующий токен после seq
    """
    def __init__(self, n_samples: int, seq_len: int, vocab_size: int, step: int = 1):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.step = step

        # заранее сгенерим данные, чтобы было проще
        starts = torch.randint(0, vocab_size, (n_samples,))
        self.x = torch.empty((n_samples, seq_len), dtype=torch.long)
        self.y = torch.empty((n_samples,), dtype=torch.long)

        for i in range(n_samples):
            s = starts[i].item()
            seq = [(s + self.step*j) % vocab_size for j in range(seq_len + 1)]
            self.x[i] = torch.tensor(seq[:seq_len], dtype=torch.long)
            self.y[i] = torch.tensor(seq[seq_len], dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# -----------------------------
# 2) Positional Encoding (самое простое синус/кос)
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.register_buffer("pe", pe)  # не обучается

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# -----------------------------
# 3) Мини-Transformer Encoder + causal mask
# -----------------------------
class TinyTransformerNextToken(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,   # важно: вход [B, T, D]
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # предсказываем распределение по vocab для "следующего токена"
        self.head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, T: int, device):
        # True = запрещено (masked)
        # shape: [T, T]
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x):
        """
        x: [B, T] long
        return logits_next: [B, V] — прогноз следующего токена
        """
        B, T = x.shape
        h = self.tok_emb(x) * math.sqrt(self.d_model)  # [B, T, D]
        h = self.pos_enc(h)
        h = self.drop(h)

        mask = self._causal_mask(T, x.device)
        h = self.encoder(h, mask=mask)  # [B, T, D]

        # берём скрытое состояние последнего токена и предсказываем следующий
        last = h[:, -1, :]              # [B, D]
        logits = self.head(last)        # [B, V]
        return logits


# -----------------------------
# 4) Train loop
# -----------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    vocab_size = 50
    seq_len = 12
    train_ds = NextTokenDataset(n_samples=20000, seq_len=seq_len, vocab_size=vocab_size, step=3)
    val_ds   = NextTokenDataset(n_samples=2000,  seq_len=seq_len, vocab_size=vocab_size, step=3)

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = TinyTransformerNextToken(
        vocab_size=vocab_size, d_model=64, nhead=4, num_layers=2,
        dim_ff=128, max_len=128, dropout=0.1
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    def eval_acc():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        model.train()
        return correct / total

    for epoch in range(1, 11):
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        acc = eval_acc()
        print(f"epoch {epoch:02d} | loss={running/len(train_dl):.4f} | val_acc={acc:.4f}")

    # -----------------------------
    # 5) Пример инференса
    # -----------------------------
    model.eval()
    x, y = val_ds[0]
    logits = model(x.unsqueeze(0).to(device))
    pred = int(logits.argmax(dim=-1).cpu().item())
    print("\nExample:")
    print("input:", x.tolist())
    print("true next:", int(y.item()))
    print("pred next:", pred)


if __name__ == "__main__":
    train()