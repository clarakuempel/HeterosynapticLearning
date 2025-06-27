"""
Code from -> https://github.com/linclab/EG_optimiser/blob/main/optim_eg/sgd_eg.py#L143

Adapted for this project
"""
import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, required)
from typing import List, Optional

from logger import log_update

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

    def __init__(self, params, lr=required, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False, update_alg="gd", block_size=4, alpha=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if update_alg not in ["gd", "md"]:
            raise ValueError("Invalid update_alg value: {}".format(update_alg))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")


        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        update_alg=update_alg, block_size=block_size, alpha=alpha)

        super(HP_SGD, self).__init__(params, defaults)

        # Attach Respective Hessian to each param group
        for group in self.param_groups:
            group['hessian'] = block_hessian(num_blocks=1, block_size=block_size, alpha=alpha)

    def __setstate__(self, state):
        super(HP_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None, log=False):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            log (bool, optional): If True, logs the update to the logger.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            # this is from optim.sgd._init_group
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            param_update = sgd(params=params_with_grad,
                grads=d_p_list,
                momentum_buffers=momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                update_alg=group['update_alg'],
                block_size=group['block_size'],
                alpha=group['alpha'],
                hessian=group['hessian']
                )

            if log and param_update is not None:
                for p, update in zip(params_with_grad, param_update):
                    name = self.param_to_name.get(p, "unknown")
                    log_update(update, name)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params: List[Tensor],
        grads: List[Tensor],
        momentum_buffers: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        update_alg: str,
        block_size: int,
        alpha: float,
        hessian,
        ):

    # Loop over the parameters
    param_updates = []
    for param_idx, param in enumerate(params):
        grad = grads[param_idx]
        if weight_decay != 0 and update_alg == "gd":
            grad = grad.add(param.sign(), alpha=weight_decay)

        if momentum != 0:
            momentum_buf = momentum_buffers[param_idx]

            if momentum_buf is None:
                momentum_buf = torch.clone(grad).detach()
                momentum_buffers[param_idx] = momentum_buf
            else:
                momentum_buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(momentum_buf, alpha=momentum)
            else:
                grad = momentum_buf


        if update_alg == 'gd':
            param.add_(grad, alpha=-lr)
            param_updates.append(grad * -lr)
        elif update_alg == "md":
            # If the parameter is a bias do a normal gradient descent step
            if param.ndim == 1:
                param.add_(grad, alpha=-lr)
                param_updates.append(grad * -lr)
            else:
                param_update = mirror_descent_update(param, grad, hessian, block_size, lr)

                param_updates.append(param_update)

    return param_updates if param_updates else None


def mirror_descent_update(param, grad, hessian, block_size, lr):
    """Mirror descent update with block-diagonal Hessian. In place operation"""
    g = grad.flatten()
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
    return update.reshape_as(param) * -lr
