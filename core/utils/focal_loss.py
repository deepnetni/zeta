import torch


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.6, reduction="elementwise_mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, target, pt):
        # pt: preidict true
        # pt = torch.sigmoid(_input)
        # pt = _input

        pt = torch.clamp(pt, min=1e-7, max=1 - 1e-7)

        alpha = self.alpha
        loss = -alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (
            1 - alpha
        ) * pt**self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == "elementwise_mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss
