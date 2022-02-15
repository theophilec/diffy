"""
Implements generating Nystorm points (on a grid).
"""
import torch


def compute_nystrom_grid(n_per_dim: int, dims: int, min_value: float, max_value: float, dtype) -> torch.Tensor:
    """
    Generate Nystrom points grid in several dimensions.

    Parameters
    ----------
    n_per_dim: int
        Number of Nyström points per dimension.
    dims: int
        Number of dimensions.
    min_value: float
        Lower bound for grid.
    max_value: float
        Upper bound for grid (with `np.linspace` conventions).
    dtype: object that is understood by `dtype=dtype` in `torch`
        Usually `torch.float32` or `torch.float64`.

    Returns
    -------
    torch.Tensor, Nystrom points grid
    """
    nystrom_out = torch.meshgrid(*[torch.linspace(min_value, max_value, n_per_dim, dtype=dtype) for _ in range(dims)])
    nystrom_out = torch.cat([grid.flatten().unsqueeze(-1) for grid in nystrom_out], dim=-1)

    return nystrom_out
