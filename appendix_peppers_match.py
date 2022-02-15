import math

import kornia.augmentation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from kornia import tensor_to_image as t2im, image_to_tensor as im2t

from did import kernels, masking
from did import nystrom
from did.dissimilarity import NormalizedDIDEstimator, compute_h, compute_q
from did.utils.imagenet import IMAGENET_NORMALIZE

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

dtype=torch.float64

mean = torch.Tensor(IMAGENET_NORMALIZE.mean)
std = torch.Tensor(IMAGENET_NORMALIZE.std)
UNORMALIZE = torchvision.transforms.Normalize((- mean / std).tolist(), (1.0 / std).tolist())

def load_peppers():
    import scipy.io
    mat = scipy.io.loadmat('data/peppers.mat')['b']
    return mat


n_rep = 20
for i in range(n_rep):
    alpha = np.random.uniform(0, 90)
    # image_size =
    # select f and g

    full_im = load_peppers() / 255.

    im_shape = full_im.shape[0:2]

    # plt.clf(); plt.imshow(full_im); ax = plt.gca(); plt.title(f"{full_im.shape} {im2t(full_im).shape}"); plt.show()

    big_h = 250
    big_start_x = np.random.randint(0, im_shape[0] - big_h)
    big_start_y = np.random.randint(0, im_shape[1] - big_h)
    big = full_im[big_start_x:big_start_x + big_h, big_start_y:big_start_y + big_h, :]
    rotated_big = kornia.geometry.rotate(im2t(big).unsqueeze(0), torch.tensor([45.]).to(dtype=torch.float64)) \
        .squeeze(0)
    small_h = int(big_h / math.sqrt(2))
    # small_center is the largest possible square in rotated_big (w/o black borders)
    small_center = rotated_big[:, big_h // 2 - small_h // 2: big_h // 2 + small_h // 2,
                   big_h // 2 - small_h // 2: big_h // 2 + small_h // 2]
    offset = 20
    small_other = im2t(
        big[offset + big_h // 2 - small_h // 2:offset + big_h // 2 + small_h // 2,
        offset + big_h // 2 - small_h // 2:offset + big_h // 2 + small_h // 2, :]
    )
    scale_factor = 1.5
    small_other_scaled = kornia.geometry.transform.scale(small_other.unsqueeze(0),
                                                         torch.tensor([scale_factor], dtype=torch.float64)).squeeze(0)

    f_image = small_other_scaled
    assert isinstance(f_image, torch.Tensor)
    g_image = small_center
    assert isinstance(g_image, torch.Tensor)
    assert f_image.shape == g_image.shape
    print(f_image.shape)

    image_size = 150
    f_image = f_image[:, :150, :150]
    g_image = g_image[:, :150, :150]

    dtype = torch.float64
    device = "cuda" if torch.cuda.is_available() else "cpu"


    f_image = IMAGENET_NORMALIZE(f_image)
    g_image = IMAGENET_NORMALIZE(g_image)

    f_image = f_image.to(dtype=dtype, device=device)
    g_image = g_image.to(dtype=dtype, device=device)

    f_coords = nystrom.compute_nystrom_grid(image_size, 2, -1, 1, dtype).to(device)
    m_x_side = 10
    m_x = m_x_side ** 2
    m_y_per_channel = 16
    n_channels = 3
    in_cholesky_reg = 1e-13
    out_cholesky_reg = 1e-13

    # Pre-computing Nystrom points
    in_nystrom = nystrom.compute_nystrom_grid(m_x_side, 2, -1, 1, dtype).to(device)
    out_nystrom = nystrom.compute_nystrom_grid(m_y_per_channel, n_channels, 0, 1, dtype).to(device)
    # normalize sampled nystrom points like images are
    out_nystrom = IMAGENET_NORMALIZE(out_nystrom.T.unsqueeze(-1)).T.squeeze().to(device)

    mask_image = masking.generate_mu_pt(f_image).to(dtype=dtype).to(device)
    mask = mask_image.flatten().unsqueeze(-1).to(device)
    # plt.imshow(t2im(mask_image)); plt.title("mask"); plt.colorbar(); plt.show();
    # plt.imshow(t2im(UNORMALIZE(f_image))); plt.title("f"); plt.show()
    # plt.imshow(t2im(UNORMALIZE(g_image))); plt.title("g"); plt.show()

    in_params = {"sigma": 1 / 6.0}
    in_kernel = kernels.Kernel("gaussian", kernels.gaussian_kernel, in_params)
    out_params = {"a": 5.0}
    out_kernel = kernels.Kernel("abel", kernels.abel_kernel, out_params)

    estimator = NormalizedDIDEstimator(in_kernel, out_kernel, in_nystrom, out_nystrom)
    estimator = estimator.pre_compute(in_cholesky_reg)

    f = f_image.permute(1, 2, 0).flatten(0, 1)
    g = g_image.permute(1, 2, 0).flatten(0, 1)

    lambda_out = 1e-2

    D_fg, eigen_h = estimator(f_coords, f, mask, f_coords, g, lambda_out)
    h = compute_h(in_kernel, f_coords, in_nystrom, estimator.r_in_inv, torch.from_numpy(eigen_h).to(device))
    q = compute_q(in_kernel, f_coords, in_nystrom, estimator.r_in_inv, torch.from_numpy(eigen_h).to(device),
                  estimator.f_hat, estimator.g_hat, estimator.g_g_inv)

    h_image = h.unflatten(0, (image_size, image_size)).unsqueeze(0)
    q_image = q.unflatten(0, (image_size, image_size)).unsqueeze(0)

    h = torch.abs(h) / torch.abs(h).max()
    q = torch.abs(q) / torch.abs(q).max()
    h_image = h.unflatten(0, (image_size, image_size)).unsqueeze(0)
    q_image = q.unflatten(0, (image_size, image_size)).unsqueeze(0)

    plt.clf()
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 2.5))
    ax[0].imshow(t2im(UNORMALIZE(f_image)))
    ax[0].set_axis_off()
    ax[0].set_title(r"$f$")
    ax[1].imshow(t2im(UNORMALIZE(f_image) * h_image))
    ax[1].set_axis_off()
    ax[1].set_title(r"$f \times h$")
    ax[2].imshow(t2im(UNORMALIZE(g_image) * q_image))
    ax[2].set_axis_off()
    ax[2].set_title(r"$g\times q$")
    ax[3].imshow(t2im(UNORMALIZE(g_image)))
    ax[3].set_axis_off()
    ax[3].set_title(r"$g$")
    fig.suptitle(fr"D(f, transform(f))= {D_fg:.2e} for $\lambda=$ {lambda_out} {i}")
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    SAVE = False
    if not SAVE:
        plt.show()
    else:
        filename = f"appendix_match_{lambda_out}_{i}"
        plt.savefig('icml_figures/' + filename + ".pdf")

print("Done")