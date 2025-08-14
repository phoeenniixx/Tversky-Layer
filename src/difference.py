import torch
import torch.nn as nn

class Difference(nn.Module):
    def __init__(self, feats: torch.Tensor, mode: str = "ignorematch"):
        super().__init__()
        self.feats = feats
        self.mode = mode

    def ignorematch(self, a:torch.Tensor, b:torch.Tensor):
        # (B, D) @ (D, K) -> (B, K)
        a_fk = torch.matmul(a, self.feats.T)
        # (P, D) @ (D, K) -> (P, K)
        b_fk = torch.matmul(b, self.feats.T)

        a_fk = a_fk.unsqueeze(1)  # (B, 1, K)
        b_fk = b_fk.unsqueeze(0)  # (1, P, K)
        mask = ((a_fk > 0) & (b_fk <= 0)).float()

        result = torch.sum(a_fk * mask, dim=2)
        return result

    def substractmatch(self, a:torch.Tensor, b:torch.Tensor):
        # (B, D) @ (D, K) -> (B, K)
        a_fk = torch.matmul(a, self.feats.T)
        # (P, D) @ (D, K) -> (P, K)
        b_fk = torch.matmul(b, self.feats.T)

        a_fk = a_fk.unsqueeze(1)  # (B, 1, K)
        b_fk = b_fk.unsqueeze(0)  # (1, P, K)
        mask = ((a_fk > 0) & (b_fk > 0) & (a_fk>b_fk)).float()

        a_masked = a_fk * mask
        b_masked = b_fk * mask

        result = torch.sum(a_masked - b_masked, dim=2)
        return result

    def forward(self, a:torch.Tensor, b:torch.Tensor):
        if self.mode =="ignorematch":
            return self.ignorematch(a, b)
        elif self.mode == "substractmatch":
            return self.substractmatch(a, b)
        else:
            raise ValueError(
                "Invalid method. Choose from 'ignorematch' or 'substractmatch'.")