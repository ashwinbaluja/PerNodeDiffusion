import torch
from torch import nn
import torch.nn.functional as F


# from https://github.com/tanelp/tiny-diffusion
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        assert size % 2 == 0 and size > 2, "size must be even and greater than 0"
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class MolecularEmbedding(nn.Module):
    def __init__(self, max_element: int, embedding_dim: int):
        super().__init__()
        self.max_element = max_element

    def forward(self, x):
        return F.one_hot(x, num_classes=self.max_element).float()
