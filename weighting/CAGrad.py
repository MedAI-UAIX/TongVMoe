import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize

from abstract_weighting import AbsWeighting


class CAGrad(AbsWeighting):
    """Conflict-Averse Gradient descent (CAGrad) for Multi-task Learning.

    Reference:
        Conflict-Averse Gradient Descent for Multi-task learning (NeurIPS 2021)

    Args:
        calpha (float): Hyperparameter controlling convergence rate
        rescale (int): Gradient rescaling type (0, 1, or 2)
    """

    def __init__(self):
        super(CAGrad, self).__init__()

    def backward(self, losses, **kwargs):
        """Apply CAGrad weighting strategy.

        Args:
            losses: Task losses tensor
            **kwargs: Contains 'calpha' and 'rescale' parameters
        """
        calpha = kwargs.get('calpha', 0.5)
        rescale = kwargs.get('rescale', 1)

        # Compute gradients for each task
        self._compute_grad_dim()
        grads = self._compute_grad(losses, mode='backward')

        # Compute gradient similarity matrix
        GG = torch.matmul(grads, grads.t()).cpu()
        g0_norm = (GG.mean() + 1e-8).sqrt()

        # Optimization setup
        x_start = np.ones(self.task_num) / self.task_num
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})

        A = GG.numpy()
        b = x_start.copy()
        c = (calpha * g0_norm + 1e-8).item()

        def objfn(x):
            """Objective function for optimization."""
            return (x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1)) +
                    c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)).sum()

        # Solve optimization problem
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(self.device)

        # Compute weighted gradient
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(0) + lmbda * gw

        # Apply rescaling
        if rescale == 0:
            new_grads = g
        elif rescale == 1:
            new_grads = g / (1 + calpha ** 2)
        elif rescale == 2:
            new_grads = g / (1 + calpha)
        else:
            raise ValueError(f'Unsupported rescale type: {rescale}')

        self._reset_grad(new_grads)
        return w_cpu