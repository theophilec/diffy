"""
Implements `DIDEstimator` and `NormalizedDIDEstimator` classes to compute the DID dissimilarity
in an efficient way. Also implements functions for computing `h` and `q` from the precomputed objects.
"""
import numpy as np
import torch

from .operators import compute_operator_from_precomputed, compute_kernel_cholesky


class DIDEstimator(torch.nn.Module):
    """
    Implementsa class.
    """

    def __init__(self, in_kernel, out_kernel, in_nystrom, out_nystrom):
        super().__init__()

        self.g_hat = None
        self.f_hat = None
        self.in_kernel = in_kernel
        self.out_kernel = out_kernel
        self.in_nystrom = in_nystrom
        self.out_nystrom = out_nystrom

        # attributes which require precomputation
        self.r_in_inv = None
        self.r_out_inv_t = None

    def pre_compute(self, reg_cholesky_in):
        r_in = compute_kernel_cholesky(self.in_nystrom, self.in_kernel, reg_cholesky_in)
        r_out = compute_kernel_cholesky(self.out_nystrom, self.out_kernel, reg_cholesky_in)
        self.r_in_inv = torch.inverse(r_in)
        self.r_out_inv_t = torch.inverse(r_out).T
        return self

    def forward(self, in_f, out_f, mask, in_g, out_g, lambda_):
        self.f_hat = compute_operator_from_precomputed(in_f, out_f, self.r_in_inv, self.r_out_inv_t, self.in_nystrom,
                                                  self.out_nystrom, mask, self.in_kernel, self.out_kernel)
        g_mask = torch.ones((out_g.shape[0], 1), dtype=mask.dtype, device=mask.device)
        self.g_hat = compute_operator_from_precomputed(in_g, out_f, self.r_in_inv, self.r_out_inv_t, self.in_nystrom,
                                                  self.out_nystrom, g_mask, self.in_kernel, self.out_kernel)
        return naive_solve(self.f_hat, self.g_hat, lambda_)


class NormalizedDIDEstimator(DIDEstimator):

    def __init__(self, in_kernel, out_kernel, in_nystrom, out_nystrom):
        super().__init__(in_kernel, out_kernel, in_nystrom, out_nystrom)
        self.f_hat = None
        self.g_hat = None
        self.g_g_inv = None
        self.f_gg_inv_f = None

    def forward(self, in_f, out_f, mask, in_g, out_g, lambda_):
        f_hat = compute_operator_from_precomputed(in_f, out_f, self.r_in_inv, self.r_out_inv_t, self.in_nystrom,
                                                       self.out_nystrom, mask, self.in_kernel, self.out_kernel)
        self.f_hat = f_hat / np.linalg.norm(f_hat.cpu().numpy(), ord=2)

        g_mask = torch.ones((out_g.shape[0], 1), dtype=mask.dtype, device=mask.device)
        g_hat = compute_operator_from_precomputed(in_g, out_g, self.r_in_inv, self.r_out_inv_t, self.in_nystrom,
                                                       self.out_nystrom, g_mask, self.in_kernel, self.out_kernel)
        self.g_hat = g_hat / np.linalg.norm(g_hat.cpu().numpy(), ord=2)
        return self.naive_solve_with_vector(lambda_)


    def naive_solve_with_vector(self, lambda_):
        m_y = self.g_hat.shape[0]
        self.g_g_inv = torch.inverse(self.g_hat @ self.g_hat.T + lambda_ * torch.eye(m_y, device=self.g_hat.device))
        self.f_gg_inv_f = self.f_hat.T @ self.g_g_inv @ self.f_hat
        eigvals, eigvecs = np.linalg.eigh(self.f_gg_inv_f.cpu().numpy())  # in ascending order
        eigval = eigvals[-1]
        eigvec = eigvecs[:, -1]
        return lambda_ * eigval, eigvec


def naive_solve(f_hat, g_hat, lambda_):
    m_y = g_hat.shape[0]
    g_g_inv = torch.inverse(g_hat @ g_hat.T + lambda_ * torch.eye(m_y, device=g_hat.device))
    f_gg_inv_f = f_hat.T @ g_g_inv @ f_hat
    return lambda_ * np.max(np.linalg.eigvalsh(f_gg_inv_f.cpu().numpy()))




def compute_h(kernel, coords, x_nystrom_points, r_in_inv, eigenvector):
    """
    Compute h map defined below at image coordinates.
    """
    return kernel(coords, x_nystrom_points) @ r_in_inv @ eigenvector


def compute_q(kernel, coords, x_nystrom_points, r_in_inv, eigenvector, f_hat, g_hat, gg_inv):
    return kernel(coords, x_nystrom_points) @ r_in_inv @ g_hat.T @ gg_inv @ f_hat @ eigenvector



