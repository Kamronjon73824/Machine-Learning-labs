"""MLP, CNN and patch-attention models for Fashion-MNIST."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    """28x28 image -> 16 patches of 7x7 -> Linear 49->64 -> 2 encoder layers -> mean pool -> 10."""

    def __init__(self, patch=7, d_model=64, nhead=4, dim_ff=128, layers=2):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, (28 // patch) ** 2, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        b, p = x.shape[0], self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)          # (B, 1, 4, 4, 7, 7)
        x = x.reshape(b, -1, p * p)                    # (B, 16, 49)
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


def build_models():
    return {"MLP": MLP(), "CNN": CNN(), "Attention": PatchAttention()}
