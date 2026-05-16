import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, d_model: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([
            Head(d_model, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, num_heads, block_size, dropout)
        self.ffwd = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MIDITransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        end_token_id: int,
        max_new_tokens: int = 200,
        temperature: float = 1.2,
        top_k: int | None = 50,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # temperature применяется один раз
            logits = logits / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))

                # берём только top-k кандидатов
                values, indices = torch.topk(logits, k=top_k, dim=-1)

                probs = torch.softmax(values, dim=-1)

                # debug: настоящие token_id
                top_probs, top_positions = torch.topk(probs, k=min(5, top_k), dim=-1)
                print("Top probabilities:")
                for p, pos in zip(top_probs[0].tolist(), top_positions[0].tolist()):
                    token_id = indices[0, pos].item()
                    print(f"token_id={token_id}, prob={p:.4f}")

                sampled_position = torch.multinomial(probs, num_samples=1)
                next_id = indices.gather(-1, sampled_position)

            else:
                probs = torch.softmax(logits, dim=-1)

                top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
                print("Top probabilities:")
                for p, token_id in zip(top_probs[0].tolist(), top_indices[0].tolist()):
                    print(f"token_id={token_id}, prob={p:.4f}")

                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

            if next_id.item() == end_token_id:
                break

        return idx