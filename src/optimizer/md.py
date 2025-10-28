"""
Code from -> https://github.com/linclab/EG_optimiser/blob/main/optim_eg/sgd_eg.py#L143

Adapted for this project
"""
import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, required)
from typing import List, Optional

# from logger import log_update

def block_hessian(num_blocks, block_size, alpha=0.1):
    assert num_blocks == 1, "For now only 1 block is supported because of shapes in the update"
    H_blocks = []
    for _ in range(num_blocks):
        H_block = torch.full((block_size, block_size), alpha)
        H_block.fill_diagonal_(1.0)
        H_blocks.append(H_block)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.block_diag(*H_blocks).to(device)

class HP_SGD(Optimizer):
    """
    Hetero-synaptic plasticity enabled SGD

    Parameters are similar to normal SGD but include:
    - update to use a mirror descent or gradient descent update (md or gd)
    - Block size for the hessian
    - alpha, to control the strength of the coupling
    """

    def __init__(self, params, lr=required, momentum=0.0, update_alg="md", block_size=4, alpha=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if update_alg not in ["md"]:
            raise ValueError("Invalid update_alg value: {}".format(update_alg))
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum value: {momentum}")


        defaults = dict(lr=lr, momentum=momentum, update_alg=update_alg, block_size=block_size, alpha=alpha)

        super(HP_SGD, self).__init__(params, defaults)

        # Attach Respective Hessian to each param group
        for group in self.param_groups:
            group['hessian'] = block_hessian(num_blocks=1, block_size=block_size, alpha=alpha)

    def __setstate__(self, state):
        super(HP_SGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            log (bool, optional): If True, logs the update to the logger.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = [] # momentum buffers

            # this is from optim.sgd._init_group
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                    exp_avgs.append(state['exp_avg'])

            sgd(params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                lr=group['lr'],
                momentum=group['momentum'],
                block_size=group['block_size'],
                hessian=group['hessian']
                )

            for p, exp_avg in zip(params_with_grad, exp_avgs):
                self.state[p]['exp_avg'] = exp_avg


def sgd(params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: list[Tensor],
        *,
        lr: float,
        momentum: float,
        block_size: int,
        hessian,
        ):

    for param_idx, param in enumerate(params):
        grad = grads[param_idx]
        exp_avg = exp_avgs[param_idx]

        if momentum > 0:
            # m_t = Bm_{t-1} + (1-B)g_t
            exp_avg.mul_(momentum).add_(grad, alpha=1 - momentum)
            grad = exp_avg

        if param.ndim == 1:
            param.add_(grad, alpha=-lr)
        else:
            mirror_descent_update(param, grad, hessian, block_size, lr)


def mirror_descent_update(param, grad, hessian, block_size, lr):
    """Mirror descent update with block-diagonal Hessian. In place operation"""
    g = grad.flatten()
    
    # Move hessian to same device as gradients
    hessian = hessian.to(g.device)
    
    spill = g.numel() % block_size
    if spill:
        main_g, tail_g = g[:-spill], g[-spill:]
        H_inv = hessian.inverse()
        H_tail_inv = hessian[:spill, :spill].inverse()
        update = torch.cat([
            (main_g.reshape(-1, block_size) @ H_inv).flatten(),
            (tail_g.reshape(1, -1) @ H_tail_inv).flatten()
        ])
    else:
        update = (g.reshape(-1, block_size) @ hessian.inverse()).flatten()

    param.add_(update.reshape_as(param), alpha=-lr)
