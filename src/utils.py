import torch
import torch.nn.functional as F

product = lambda a, b: a * b
mins = lambda a, b: torch.minimum(a, b)
maxs = lambda a, b: torch.maximum(a, b)
mean = lambda a, b: (a + b) / 2.0
gmean = lambda a, b: torch.sqrt(torch.clamp(a, min=0) * torch.clamp(b, min=0))

def softmin(a, b, k=10.0):
    stacked_tensors = torch.stack([a, b], dim=-1)
    weights = F.softmax(-k * stacked_tensors, dim=-1)
    return torch.sum(weights * stacked_tensors, dim=-1)

