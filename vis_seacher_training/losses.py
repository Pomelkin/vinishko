import torch
import torch.nn as nn
import torch.nn.functional as F


class SubCenterArcFace(nn.Module):
    margins: torch.Tensor

    def __init__(
        self,
        dim: int,
        n_classes: int,
        class_counts: torch.Tensor,
        k: int = 3,
        s: float = 30.0,
        m_a: float = 0.5,
        m_b: float = 0.05,
        m_lambda: float = 0.25,
    ) -> None:
        super().__init__()
        self.k = k
        self.s = s
        self.weight = nn.Parameter(torch.empty(n_classes * k, dim))
        nn.init.xavier_uniform_(self.weight)
        margins = m_a * class_counts.float().pow(-m_lambda) + m_b
        self.register_buffer("margins", margins)
        return

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(emb), F.normalize(self.weight))
        cos = cos.view(emb.size(0), -1, self.k).max(dim=2).values
        m = self.margins[labels]
        theta = torch.acos(cos.gather(1, labels[:, None]).clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + m[:, None])
        logits = cos.scatter(1, labels[:, None], target)
        return F.cross_entropy(self.s * logits, labels)
