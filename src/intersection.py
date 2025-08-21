import torch
import torch.nn as nn
from .utils import *

class Intersection(nn.Module):

    def __init__(self, mode: str="product"):
        super().__init__()
        # self.feats =feats
        self.mode = mode
        if self.mode == "product":
            self.op = product
        elif self.mode == "gmean":
            self.op = gmean
        elif self.mode == "max":
            self.op = maxs
        elif self.mode == "min":
            self.op = mins
        elif self.mode == "mean":
            self.op = mean
        elif self.mode == "softmin":
            self.op = softmin
        else:
            raise ValueError(
                "Invalid method. Choose from 'product', 'gmean', 'max', 'min', 'softmin' or 'mean'.")

    def forward(self, a:torch.Tensor, b:torch.Tensor, feats: torch.Tensor):

        # (B, D) @ (D, K) -> (B, K)
        a_fk = torch.matmul(a, feats.T)
        # (P, D) @ (D, K) -> (P, K)
        b_fk = torch.matmul(b, feats.T)

        a_fk = a_fk.unsqueeze(1)  # (B, 1, K)
        b_fk = b_fk.unsqueeze(0)  # (1, P, K)
        mask = ((a_fk > 0) & (b_fk > 0)).float()

        aggregated = self.op(a_fk, b_fk)  # (B, P, K)

        result = torch.sum(aggregated * mask, dim=2)
        return result # (B, P)