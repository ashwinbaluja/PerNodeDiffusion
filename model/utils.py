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


# probably could be improved, could be redundant
# maybe just leave this to the transformer?
class MolecularEmbedding(torch.nn.Module):
    def __init__(self, max_element: int, output: int = 8):
        super().__init__()
        self.max_element = max_element
        self.hidden = torch.nn.Linear(max_element, max_element)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(max_element, output)

    def forward(self, x):
        onehot = F.one_hot(x, num_classes=self.max_element).float()
        x = self.hidden(onehot)
        x = self.relu(x)
        x = self.output(x)
        return x


# https://hunterheidenreich.com/posts/kabsch_algorithm/
def kabsch_torch_batched(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  #

    # Optimal translation
    t = centroid_Q - centroid_P  # Bx1x3
    t = t.squeeze(1)  # Bx3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(1, 2), q)  # Bx3x3

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # Bx3x3

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B

    flip = d < 0.0

    # Create a condition tensor that matches the shape of Vt but only for the last column
    condition = flip.unsqueeze(-1).expand_as(Vt).to(P.device)

    # ! edited from orig to remove inplace modification, no longer necessary as we stop_grad it
    Vt_new = torch.where(condition, Vt * torch.tensor([-1.0]).to(P.device), Vt).to(
        P.device
    )

    # Optimal rotation
    R = torch.matmul(Vt_new.transpose(1, 2), U.transpose(1, 2))

    return R, t
