import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbsWeighting(nn.Module):
    """Abstract class for multi-task learning weighting strategies.

    This class provides the basic framework for implementing various
    multi-task learning weighting methods.
    """

    def __init__(self):
        super(AbsWeighting, self).__init__()

    def init_param(self):
        """Initialize trainable parameters for specific weighting methods."""
        pass

    def _compute_grad_dim(self):
        """Compute the total dimension of gradients."""
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        """Convert gradients to a single vector."""
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count + 1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode='backward'):
        """Compute gradients for each task.

        Args:
            losses: List of task losses
            mode: Gradient computation mode ('backward' or 'autograd')
        """
        grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        for tn in range(self.task_num):
            if mode == 'backward':
                losses[tn].backward(retain_graph=True) if (tn + 1) != self.task_num else losses[tn].backward()
                grads[tn] = self._grad2vec()
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                grads[tn] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError(f'Unsupported gradient computation mode: {mode}')
            self.zero_grad_share_params()
        return grads

    def _reset_grad(self, new_grads):
        """Reset gradients with new computed gradients."""
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count + 1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1

    def _get_grads(self, losses, mode='backward'):
        """Get gradients for all tasks.

        Returns:
            Gradients tensor with shape [task_num, grad_dim]
        """
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode)
        return grads

    def _backward_new_grads(self, batch_weight, grads):
        """Apply weighted gradients.

        Args:
            batch_weight: Weight for each task
            grads: Gradients for each task
        """
        new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
        self._reset_grad(new_grads)

    def backward(self, losses, **kwargs):
        """Main backward function to be implemented by specific methods.

        Args:
            losses: List of task losses
            kwargs: Additional hyperparameters
        """
        raise NotImplementedError("Subclasses must implement the backward method")