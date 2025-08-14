import torch
import torch.nn as nn

from .similarity_layer import TverskySimilarity

class TverskyProjection(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, num_features: int,
                 intersection_mode: str = "product", difference_mode: str = "ignorematch"):
        super().__init__()
        self.input_dim = input_dim  # D
        self.output_dim = output_dim  # P
        self.num_features = num_features  # K
        self.intersection_mode = intersection_mode
        self.difference_mode = difference_mode

        self.features = nn.Parameter(torch.randn(self.num_features, self.input_dim),
                                     requires_grad=True)
        self.projections = nn.Parameter(torch.randn(self.output_dim, self.input_dim),
                                        requires_grad=True)
        self.similarity_calculator = TverskySimilarity(self.features,
                                                       intersection_mode=self.intersection_mode,
                                                       difference_mode=self.difference_mode)

    def forward(self, x:torch.Tensor):
         return self.similarity_calculator(x, self.projections)


