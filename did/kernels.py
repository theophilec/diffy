"""
kernels.py

Implements generic Kernel module as well as Abel and Gaussian kernels.
"""
import torch


class Kernel(torch.nn.Module):
    """Generic Kernel nn.Module.

    Implements a `validate_input` static method and forward.
    """

    def __init__(self, kernel_name, kernel_fn, kernel_params):
        super().__init__()
        self.kernel_name = kernel_name
        self.kernel_fn = kernel_fn
        self.kernel_params = kernel_params

    @staticmethod
    def validate_input(x):
        """Validate input, else add dimension in front.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor to validate.

        Returns
        -------
        torch.Tensor
            Validated input tensor, or unsqueezed at 0 dimension.
        """
        if x.ndim == 1:
            return x.unsqueeze(0)
        else:
            return x

    def forward(self, x, y):
        x = Kernel.validate_input(x)
        y = Kernel.validate_input(y)
        return self.kernel_fn(x, y, **self.kernel_params)


def sq_dist(X1, X2):
    """
    Wrapper for `torch.cdist` with exponent 2.
    
    :math:`\Vert X_1 - X_2\Vert^2_2`

    Parameters
    ----------
    X1: torch.Tensor with size (n, d)
    X2: torch.Tensor with size (m, d)

    Returns
    -------
    torch.Tensor with size (n, m)
    """
    return torch.cdist(X1, X2).pow_(2)


def abel_kernel(X1, X2, a=1.0):
    """
    Kernel function for Abel kernel defined as: 

    :math:`k(x, y) = \exp(-a\Vert x - y\Vert_2)`.

    Parameters
    ----------
    X1: torch.Tensor with size (n, d)
    X2: torch.Tensor with size (m, d)
    a: float, parameter

    Returns
    -------
    torch.Tensor with size (n, m)
    """
    return sq_dist(X1, X2).sqrt_().mul_(- a).exp_()


def gaussian_kernel(X1, X2, sigma=1.0):
    """
    Kernel function for Gaussian kernel defined as: 

    
    :math:`k(x, y) = \exp\left(-\\frac{\Vert x - y\Vert_2 ^2}{2 \sigma^2}\\right)`.

    Parameters
    ----------
    X1: torch.Tensor with size (n, d)
    X2: torch.Tensor with size (m, d)
    sigma: float, bandwidth parameter

    Returns
    -------
    torch.Tensor with size (n, m)
    """
    return sq_dist(X1, X2).mul_(- 1 / (2 * sigma ** 2)).exp_()
