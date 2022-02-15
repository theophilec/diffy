import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from kornia import image_to_tensor as im2t
from kornia.geometry.transform import rotate

from did import kernels, masking as mask_
from did import nystrom
from did.dissimilarity import NormalizedDIDEstimator, compute_h, compute_q
from did.utils.imagenet import IMAGENET_NORMALIZE

SEED = 15
torch.manual_seed(SEED)
np.random.seed(SEED)

image_size = 100
H, W = image_size, image_size

mean = torch.Tensor(IMAGENET_NORMALIZE.mean)
std = torch.Tensor(IMAGENET_NORMALIZE.std)
UNORMALIZE = torchvision.transforms.Normalize((- mean / std).tolist(), (1.0 / std).tolist())


def load_peppers():
    import scipy.io
    mat = scipy.io.loadmat('data/peppers.mat')['b']
    return mat


full_im = load_peppers() / 255.

# select f and g


dtype = torch.float64
device = "cuda:2"

# send to torch
acc = []
acc2 = []
n_repeat = 500
f_image = im2t(full_im)[:, 150:250, 150:250]
f_image = IMAGENET_NORMALIZE(f_image)
f_image = f_image.to(dtype=dtype, device=device)
m_x_side = 10
m_x = m_x_side ** 2
m_y_per_channel = 16
n_channels = 3
in_cholesky_reg = 1e-13
out_cholesky_reg = 1e-13
f_coords = nystrom.compute_nystrom_grid(image_size, 2, -1, 1, dtype).to(device)

# Pre-computing Nystrom points
in_nystrom = nystrom.compute_nystrom_grid(m_x_side, 2, -1, 1, dtype).to(device)
out_nystrom = nystrom.compute_nystrom_grid(m_y_per_channel, n_channels, 0, 1, dtype).to(device)
# normalize sampled nystrom points like images are
out_nystrom = IMAGENET_NORMALIZE(out_nystrom.T.unsqueeze(-1)).T.squeeze().to(device)

mask_image = mask_.generate_mu_pt(f_image).to(dtype=dtype).to(device)
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

for _ in range(n_repeat):

    # alpha = 45.0  # deg
    g_image = im2t(full_im)
    g_image = IMAGENET_NORMALIZE(g_image)
    # g_image = rotate(g_image.unsqueeze(0), torch.tensor([alpha]).to(dtype=dtype),
    #                  torch.tensor([[(150. + 250) / 2, (150. + 250) / 2]]).to(dtype=dtype)).squeeze(0)
    size = 100
    rand_x = np.random.randint(0, full_im.shape[0] - size)
    rand_y = np.random.randint(0, full_im.shape[1] - size)
    g_image = g_image[:, rand_x:rand_x + size, rand_y:rand_y + size]
    # g_image = g_image[:, 250:350, 150:250]

    g_image = g_image.to(dtype=dtype, device=device)

    g = g_image.permute(1, 2, 0).flatten(0, 1)

    lambda_out = 1e-2

    # note that we assume that f and g have the same coords here (can be changed)
    D_fg, eigen_h = estimator(f_coords, f, mask, f_coords, g, lambda_out)


    # h = compute_h(in_kernel, f_coords, in_nystrom, estimator.r_in_inv, torch.from_numpy(eigen_h).to(device))
    # q = compute_q(in_kernel, f_coords, in_nystrom, estimator.r_in_inv, torch.from_numpy(eigen_h).to(device),
    #               estimator.f_hat, estimator.g_hat, estimator.g_g_inv)
    # h_image = h.unflatten(0, (image_size, image_size)).unsqueeze(0)

    def rescale(t):
        return (t - t.min()) / (t.max() - t.min())


    # plt.imshow(t2im(UNORMALIZE(f_image)), alpha=0.3)
    # plt.imshow(t2im(rescale(h_image)),cmap='gray', alpha=0.5)
    # plt.title("h")
    # plt.colorbar();
    # plt.show()
    # q_image = q.unflatten(0, (image_size, image_size)).unsqueeze(0)
    l2 = torch.norm(g_image - f_image)
    if _ % 100 == 0:
        print(f"{_} / {n_repeat}")
        print(f"D(f, g) = {D_fg} | l2 = {l2} | lambda_out = {lambda_out}")
    # plt.imshow(t2im(UNORMALIZE(g_image)), alpha=0.3)
    # plt.imshow(t2im(rescale(q_image)),cmap='gray', alpha=0.5)
    # plt.colorbar();
    # plt.title(f"q. lambda_out = {lambda_out:.2e} | D(f, g) = {D_fg:.3e} | L2(f, g) = {torch.norm(g_image - f_image)}")
    # plt.show()
    acc.append(D_fg)
    acc2.append(l2.cpu().item())

# l2_ = [a / max(acc2)  for a in acc2]
l2_ = np.asarray(acc2)
# diff_ = [a / max(acc) for a in acc]
diff_ = np.asarray(acc)

acc = []
acc2 = []
alphas = np.linspace(0.0, 180, 10)
for alpha in alphas:
    # alpha = 45.0  # deg
    f_image = im2t(full_im)[:, 150:250, 150:250]
    f_image = IMAGENET_NORMALIZE(f_image)
    g_image = im2t(full_im)
    g_image = IMAGENET_NORMALIZE(g_image)
    g_image = rotate(g_image.unsqueeze(0), torch.tensor([alpha]).to(dtype=dtype),
                     torch.tensor([[(150. + 250) / 2, (150. + 250) / 2]]).to(dtype=dtype)).squeeze(0)
    g_image = g_image[:, 150:250, 150:250]
    # g_image = g_image[:, 250:350, 150:250]

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

    mask_image = mask_.generate_mu_pt(f_image).to(dtype=dtype).to(device)
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

    # note that we assume that f and g have the same coords here (can be changed)
    D_fg, eigen_h = estimator(f_coords, f, mask, f_coords, g, lambda_out)
    h = compute_h(in_kernel, f_coords, in_nystrom, estimator.r_in_inv, torch.from_numpy(eigen_h).to(device))
    q = compute_q(in_kernel, f_coords, in_nystrom, estimator.r_in_inv, torch.from_numpy(eigen_h).to(device),
                  estimator.f_hat, estimator.g_hat, estimator.g_g_inv)
    h_image = h.unflatten(0, (image_size, image_size)).unsqueeze(0)


    def rescale(t):
        return (t - t.min()) / (t.max() - t.min())


    # plt.imshow(t2im(UNORMALIZE(f_image)), alpha=0.3)
    # plt.imshow(t2im(rescale(h_image)),cmap='gray', alpha=0.5)
    # plt.title("h")
    # plt.colorbar();
    # plt.show()
    q_image = q.unflatten(0, (image_size, image_size)).unsqueeze(0)
    l2 = torch.norm(g_image - f_image)
    print(f"D(f, g) = {D_fg} | l2 = {l2} | lambda_out = {lambda_out}")
    # plt.imshow(t2im(UNORMALIZE(g_image)), alpha=0.3)
    # plt.imshow(t2im(rescale(q_image)),cmap='gray', alpha=0.5)
    # plt.colorbar();
    # plt.title(f"q. lambda_out = {lambda_out:.2e} | D(f, g) = {D_fg:.3e} | L2(f, g) = {torch.norm(g_image - f_image)}")
    # plt.show()
    acc.append(D_fg)
    acc2.append(l2.cpu().item())

l2_rotation = [a for a in acc2]
diff_rotation = [a for a in acc]
print("done")

beta = 0.10
diff_lower = np.quantile(diff_, beta / 2)
diff_higher = np.quantile(diff_, 1 - beta / 2)
l2_lower = np.quantile(l2_, beta / 2)
l2_higher = np.quantile(l2_, 1 - beta / 2)
diff_mean = diff_.mean()
diff_std = diff_.std()
l2_mean = l2_.mean()
l2_std = l2_.std()

plt.scatter(alphas, diff_rotation, marker='+', c='k')
plt.scatter(alphas[1:-1], l2_rotation[1:-1], marker='x', c='k')
plt.fill_between(alphas, diff_mean - diff_std, diff_mean + diff_std, color="red", ls=':', alpha=0.3)
plt.fill_between(alphas, l2_mean - l2_std, l2_mean + l2_std, color="blue", ls='--', alpha=0.3)
plt.xscale("log")
plt.yscale("log")
plt.show()

SAVE = False
plt.scatter(alphas, diff_rotation, marker='+', c='k')
plt.scatter(alphas[1:-1], l2_rotation[1:-1], marker='x', c='k')
plt.fill_between(alphas, diff_lower, diff_higher, color="red", ls=':', alpha=0.3)
plt.fill_between(alphas, l2_lower, l2_higher, color="blue", ls='--', alpha=0.3)
# plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
if not SAVE:
    plt.show()
else:
    plt.savefig("icml_figures/rotation_points.svg")
