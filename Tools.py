from torch import nn
import torch


def to_onehot(indices, num_classes):
    new_shape = (indices.shape[0], num_classes) + indices.shape[1:]
    onehot = torch.zeros(new_shape, dtype=torch.uint8, device=indices.device)
    return onehot.scatter_(1, indices.unsqueeze(1), 1)


class Precision(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.tp = torch.zeros(num_classes)
        self.total = torch.zeros(num_classes)

        self._eps = 1e-6

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        y = to_onehot(targets.long(), num_classes=self.num_classes)
        pred_indices = torch.argmax(preds, dim=1)
        y_pred = to_onehot(pred_indices.view(-1), num_classes=self.num_classes)
        self.tp += (y_pred * y).sum(dim=0)
        self.total += y_pred.sum(dim=0)

    def compute(self) -> torch.Tensor:
        return (self.tp / (self.total + self._eps)).mean()

    def clear(self) -> None:
        self.tp = torch.zeros(num_classes)
        self.total = torch.zeros(num_classes)


class Recall(Precision):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        y = to_onehot(targets.long(), num_classes=self.num_classes)
        pred_indices = torch.argmax(preds, dim=1)
        y_pred = to_onehot(pred_indices.view(-1), num_classes=self.num_classes)
        self.tp += (y_pred * y).sum(dim=0)
        self.total += y.sum(dim=0)
