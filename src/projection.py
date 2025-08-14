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
        if x.dim() == 2:
            output = self.similarity_calculator(x, self.prototypes)
        elif x.dim() == 3:
            batch_size, time_steps, input_dim = x.shape
            x_flat = x.reshape(-1, input_dim)
            similarity_flat = self.similarity_calculator(x_flat, self.projections)
            output = similarity_flat.reshape(batch_size, time_steps, self.output_dim)
        else:
            raise ValueError(
                f"TverskyProjection received an unsupported input tensor shape. "
                f"Expected 2D (batch, features) or 3D (batch, time, features), but got {x.dim()}D."
            )
        return output


