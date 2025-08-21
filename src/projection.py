import torch
import torch.nn as nn

from .similarity_layer import TverskySimilarity

class TverskyProjection(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, num_features: int,
                 intersection_mode: str = "product", difference_mode: str = "ignorematch",
                 output_scale: float = 0.01, use_batch_norm: bool = True,
                 init_std: float = 0.01):
        super().__init__()
        self.input_dim = input_dim  # D
        self.output_dim = output_dim  # P
        # self.num_features = num_features  # K
        self.intersection_mode = intersection_mode
        self.difference_mode = difference_mode
        self.output_scale = output_scale
        self.use_batch_norm = use_batch_norm
        self.init_std = init_std

        # self.features = nn.Parameter(torch.randn(self.num_features, self.input_dim),
        #                              requires_grad=True)
        self.feature_generator = nn.Linear(input_dim, num_features * input_dim)
        self.projections = nn.Parameter(torch.randn(self.output_dim, self.input_dim),
                                        requires_grad=True)
        self.similarity_calculator = TverskySimilarity(self.features,
                                                       intersection_mode=self.intersection_mode,
                                                       difference_mode=self.difference_mode)

        self._initialize_weights()

        if self.use_batch_norm and self.output_dim > 1:
            self.output_norm = nn.BatchNorm1d(self.output_dim)
        else:
            self.output_norm = None

    def _apply_output_normalization(self, output):
        """Apply batch normalization if enabled"""
        if self.output_norm is not None and len(output.shape) == 3:
            output_reshaped = output.transpose(1, 2)
            output_normed = self.output_norm(output_reshaped)
            output = output_normed.transpose(1, 2)
        elif self.output_norm is not None and len(output.shape) == 2:
            output = self.output_norm(output)
        return output

    def _initialize_weights(self):
        """Initialize with much smaller, more stable weights"""
        nn.init.normal_(self.features, mean=0, std=self.init_std)
        nn.init.normal_(self.projections, mean=0, std=self.init_std)

        with torch.no_grad():
            nn.init.uniform_(self.similarity_calculator.constants, -0.1, 0.1)

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

        output = output * self.output_scale
        output = self._apply_output_normalization(output)
        return output


