"""
Implements methods for computing approximate operators for the DID computation.
"""
import torch

from .kernels import Kernel


def compute_kernel_cholesky(nystrom_points: torch.Tensor, kernel: Kernel, cholesky_reg: float):
    """
    Compute Cholesky decomposition of the kernel matrix for Nystrom points,
    i.e. `sqrt(K(X_tilde, X_tilde))`.

    Parameters
    ----------
    nystrom_points: torch.Tensor 
        Shape (n, d)
    kernel: Kernel or callable
        Kernel on the space
    cholesky_reg: float
        Regularization parameter for Cholesky decomposition.

    Returns
    -------
    torch.Tensor, (n, n)

    """
    device = nystrom_points.device
    n_nystrom_points = nystrom_points.shape[0]
    kernel_nystrom_nystrom = kernel(nystrom_points, nystrom_points)
    chol = torch.cholesky(kernel_nystrom_nystrom + cholesky_reg * torch.eye(n_nystrom_points, device=device), upper=True)
    return chol


def compute_operator(in_nystrom: torch.Tensor, in_points: torch.Tensor, in_cholesky_reg: float,
                     out_nystrom: torch.Tensor,
                     out_points: torch.Tensor, out_cholesky_reg: float, in_kernel: Kernel, out_kernel: Kernel,
                     mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the Nystrom approximation of the operator (without projection operator).

    This method gives full implementation (without precomputation). See `compute_operator_with_precomputation`.

    Parameters
    ----------
    in_nystrom: torch.Tensor
        Nystrom points on the input space (X).
    in_points: torch.Tensor
        Coordinates of the sample points.
    in_cholesky_reg: float
        Regularization parameter for Cholesky decomposition on input.
    out_nystrom: torch.Tensor
        Nystrom points on the output space (Y).
    out_points: torch.Tensor
        Values taken by the function.
    out_cholesky_reg: float
        Regularization parameter for Cholesky decomposition on input.
    in_kernel: Kernel or callable
        Kernel on the input space (X).
    out_kernel: Kernel or callable
        Kernel on the output space (Y).
    mask: torch.Tensor
        Mask function at sample points (mu).

    Returns
    -------
    F_hat, torch.Tensor
    """
    r_in = compute_kernel_cholesky(in_nystrom, in_kernel, in_cholesky_reg)
    r_in_inv = torch.inverse(r_in)
    s = in_kernel(in_points, in_nystrom)
    r_out = compute_kernel_cholesky(out_nystrom, out_kernel, out_cholesky_reg)
    r_out_inv_t = torch.inverse(r_out).T
    z = out_kernel(out_nystrom, out_points)
    n = len(in_points)
    return 1 / n * r_out_inv_t @ z @ (mask * s) @ r_in_inv


def compute_operator_from_precomputed(in_points: torch.Tensor, out_points: torch.Tensor, r_in_inv: torch.Tensor, 
        r_out_inv_t: torch.Tensor, in_nystrom: torch.Tensor, out_nystrom: torch.Tensor, mask: torch.Tensor, 
        in_kernel: Kernel, out_kernel: Kernel):
    """
    Computes operators from common precomputed objects. Useful to avoid repeating computations.

    Parameters
    ----------
    in_points: torch.Tensor
        Coordinates of the sample points.
    out_points: torch.Tensor
        Values taken by the function at the sample points.
    r_in_inv: torch.Tensor
        Cholesky inverse of the kernel matrix over input Nyström points.
    r_out_inv_t: torch.Tensor
        Cholesky inverse and transpose of the kernel matrix over output Nyström points.
    in_nystrom: torch.Tensor
        Nystrom points on the input space.
    out_nystrom: torch.Tensor
        Nystrom points on the output space.
    mask: torch.Tensor
        Mask function at sample points (mu).
    in_kernel: Kernel or callable
        Kernel on the input space (X).
    out_kernel: Kernel or callable
        Kernel on the output space (Y).
    
    Returns
    -------
    torch.Tensor: representation of the Nyström approximation of the operator.
    """
    s = in_kernel(in_points, in_nystrom)
    z = out_kernel(out_nystrom, out_points)
    n = len(in_points)

    return 1 / n * r_out_inv_t @ z @ (mask * s) @ r_in_inv
