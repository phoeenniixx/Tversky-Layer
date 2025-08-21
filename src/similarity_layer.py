import torch

from .intersection import Intersection
from .difference import Difference
import torch.nn as nn

class TverskySimilarity(nn.Module):
    def __init__(self, intersection_mode: str = "product", difference_mode: str = "ignorematch"):
        super().__init__()
        # self.feats = feats
        self.intersection_mode = intersection_mode
        self.difference_mode = difference_mode

        initial_constants = torch.tensor([[[1.0, -1.0, -1.0]]])  # (1, 1, 3)
        self.constants = nn.Parameter(initial_constants, requires_grad=True)
        self.intersection = Intersection(mode=intersection_mode)
        self.difference = Difference(mode=difference_mode)

    def forward(self, a:torch.Tensor, b:torch.Tensor, feats:torch.Tensor):
        ab_intersec = self.intersection(a, b, feats)

        ab_diff = self.difference(a, b, feats)
        ba_diff = self.difference(b, a, feats).T
        if ba_diff.shape != ab_diff.shape:
            ba_diff = ba_diff.permute(1, 2, 0)
        stacked_features = torch.stack((ab_intersec, ab_diff, ba_diff), dim=1) # (B, 3, P)

        similarity = torch.matmul(self.constants, stacked_features) # (B, 1, P)

        return similarity.squeeze(1) # (B,P)