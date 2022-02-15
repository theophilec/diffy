import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from kornia import tensor_to_image as t2im, image_to_tensor as im2t

from data.data_loading import load_perspective, float_array_to_pil
from did import kernels, masking
from did import nystrom
from did.dissimilarity import NormalizedDIDEstimator, compute_h, compute_q
from did.utils.imagenet import IMAGENET_NORMALIZE

SEED = 15
torch.manual_seed(SEED)
np.random.seed(SEED)

image_size = 200
H, W = image_size, image_size

mean = torch.Tensor(IMAGENET_NORMALIZE.mean)
std = torch.Tensor(IMAGENET_NORMALIZE.std)
UNORMALIZE = torchvision.transforms.Normalize((- mean / std).tolist(), (1.0 / std).tolist())

scene_index = 4
full_im = load_perspective(scene_index)
full_im_array_1 = np.asarray(full_im) / 255.0
size = min(full_im_array_1.shape[0:2])
full_im_array_1 = full_im_array_1[:size, :, :]
full_im = Image.fromarray(np.uint8(full_im_array_1 * 255))
plt.imshow(full_im_array_1);
plt.grid();
plt.show();
full_im = full_im.resize((image_size, image_size), Image.LANCZOS)
full_im_array_1 = np.asarray(full_im) / 255.0
plt.imshow(full_im_array_1);
plt.grid();
plt.show();

scene_index = 1
full_im = load_perspective(scene_index)
full_im_array_2 = np.asarray(full_im) / 255.0
size = min(full_im_array_2.shape[0:2])
full_im_array_2 = full_im_array_2[:size, :, :]  # make square
print(full_im_array_2.dtype)
print(full_im_array_2.shape)
plt.imshow(full_im_array_2);
plt.grid();
plt.show();
full_im_array_2 = full_im_array_2[200:700, 100:600, :]  # select focus
full_im = float_array_to_pil(full_im_array_2)
full_im = full_im.resize((image_size, image_size), Image.LANCZOS)
# full_im = full_im.rotate(90)
full_im_array_2 = np.asarray(full_im) / 255.
plt.imshow(full_im_array_2);
plt.grid();
plt.show();

# select f and g
f_image = im2t(full_im_array_2)
g_image = im2t(full_im_array_1)

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
# plt.imshow(t2im(mask_image)); plt.title("mask"); plt.show();
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

plt.imshow(t2im(UNORMALIZE(f_image)), alpha=1)
plt.imshow(t2im(h_image >= 0.1), cmap='gray', alpha=0.5)
# plt.title("h")
ax = plt.gca()
ax.set_axis_off()
ax.margins(0, 0)
plt.tight_layout()
plt.show()

plt.imshow(t2im(UNORMALIZE(g_image)), alpha=1)
plt.imshow(t2im(q_image) >= 1., cmap='gray', alpha=0.6)
plt.xticks([])
plt.yticks([])
plt.title(f"q. lambda_out = {lambda_out:.2e} | D(f, g) = {D_fg:.3e}")
plt.tight_layout()
plt.show()
