import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbsLoss(object):
    """Abstract base class for loss functions in multi-task learning."""

    def __init__(self):
        self.record = []
        self.bs = []

    def compute_loss(self, pred, gt):
        """Calculate the loss between predictions and ground truth.

        Args:
            pred (torch.Tensor): Model predictions
            gt (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss value
        """
        raise NotImplementedError("Subclasses must implement compute_loss method")

    def _update_loss(self, pred, gt):
        """Update loss records with current batch.

        Args:
            pred (torch.Tensor): Model predictions
            gt (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Current batch loss
        """
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred.size()[0])
        return loss

    def _average_loss(self):
        """Calculate weighted average loss across all batches.

        Returns:
            float: Average loss value
        """
        if not self.record:
            return 0.0
        record = np.array(self.record)
        bs = np.array(self.bs)
        return (record * bs).sum() / bs.sum()

    def _reinit(self):
        """Reset loss records for next epoch."""
        self.record = []
        self.bs = []


class CELoss(AbsLoss):
    """Cross-entropy loss function for classification tasks."""

    def __init__(self, weight=None, reduction='mean'):
        super(CELoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def compute_loss(self, pred, gt):
        """Compute cross-entropy loss.

        Args:
            pred (torch.Tensor): Model predictions [batch_size, num_classes]
            gt (torch.Tensor): Ground truth labels [batch_size]

        Returns:
            torch.Tensor: Cross-entropy loss
        """
        return self.loss_fn(pred, gt)

