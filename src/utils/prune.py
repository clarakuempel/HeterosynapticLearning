# Implementing the NM pruning
import torch
from torch.nn.utils.prune import BasePruningMethod

class NMPruning(BasePruningMethod):
    PRUNING_TYPE = 'structured'
    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.dim = 0 # in the row dim

    def compute_mask(self, t, default_mask):
        """ Default mask is the previous mask, 1's most of the time.

            Compute mask for t
        """
        assert t.dim() == 2, f"NM pruning only supports 2D tensors, got {t.dim()}D tensor"
        assert t.shape[1] % self.M == 0, f"The second dimension ({t.shape[1]}) of the tensor must be divisible by M ({self.M})"

        mask = default_mask.clone()

        # View by blocks
        rows, cols = t.shape
        t_blocks = t.view(rows, cols // self.M, self.M)
        abs_t = t_blocks.abs()
        topk = abs_t.topk(self.N, dim=2).indices
        new_mask = torch.zeros_like(t_blocks)
        new_mask.scatter_(-1, topk, 1.0)

        return new_mask.view_as(t)



def apply_nm_pruning(model, N, M):
    """Apply NM pruning to all eligible modules in the model and print which layers are pruned."""
    print(f"Applying {N}:{M} NM pruning to model layers...")
    for name, module in model.named_modules():
        # If the module has a weight_orig then it must have a weight attribute
        # If it does not have a weight_orig, then it must have a weight Parameter

        # NOTE: We apply the new pruning to the weight attribute meaning:
        # - If the module has already been pruned we work ontop of the pruned
        #   version
        if hasattr(module, 'weight_orig') or (hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter)):
            try:
                NMPruning(N=N, M=M).apply(module, 'weight', N=N, M=M)
                print(f"  -> Pruned layer: {name}.weight")
            except Exception as e:
                print(f"  -> Skipped layer: {name} ({e})")
