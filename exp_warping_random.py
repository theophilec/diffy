import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from kornia import tensor_to_image as t2im

from did.utils import imagenet
import warping.warp as warp
from did import kernels
from did import nystrom
from did.dissimilarity import NormalizedDIDEstimator
from did.utils.imagenet import IMAGENET_NORMALIZE, load_imagenet

SEED = 13
torch.manual_seed(SEED)
np.random.seed(SEED)

image_size = 150

path_train = imagenet.PATH_TO_IMAGENET_TRAIN
path_test = imagenet.PATH_TO_IMAGENET_TEST
# image_size = 200
batchsize = 50
num_workers = 1
IMAGENETTE = True
trainset, _, _, _, n_classes = load_imagenet(path_train, path_test, image_size,
                                             batchsize, num_workers,
                                             n_classes=10 if IMAGENETTE else None)
mean = torch.Tensor(IMAGENET_NORMALIZE.mean)
std = torch.Tensor(IMAGENET_NORMALIZE.std)
UNORMALIZE = torchvision.transforms.Normalize((- mean / std).tolist(), (1.0 / std).tolist())

# Example 1: dog (1376)
image_i = 6783  # gas station
# image_i = 7235

T = 1e-3
c = 2

dtype = torch.float64
device = "cuda:2"

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

# plt.imshow(t2im(UNORMALIZE(g_image)))
# plt.title("g")
# plt.show()

in_params = {"sigma": 1 / 6.0}
in_kernel = kernels.Kernel("gaussian", kernels.gaussian_kernel, in_params)
out_params = {"a": 5.0}
out_kernel = kernels.Kernel("abel", kernels.abel_kernel, out_params)

estimator = NormalizedDIDEstimator(in_kernel, out_kernel, in_nystrom, out_nystrom)
estimator = estimator.pre_compute(in_cholesky_reg)

n_rep = 1
n_images = 1000
image_ids = np.random.randint(0, len(trainset), n_images)

lambda_out_tab = [1e-6]  # , 1e-4, 1e-3, 1e-2] #np.logspace(-10, 0, 3)
T_tab = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 10]
acc = []
l2 = []
T_tab_ = []
for i, image_id in enumerate(image_ids):
    print(i)
    im0 = trainset[image_id][0]
    f_image = im0
    f_image = f_image.to(dtype=dtype, device=device)
    if i == 0:
        mask_image = mask.generate_mu_pt(f_image).to(dtype=dtype).to(device)
        mask = mask_image.flatten().unsqueeze(-1).to(device)
    f = f_image.permute(1, 2, 0).flatten(0, 1)
    for T in T_tab:
        for irep in range(n_rep):
            g_image = im0 if T == 0 else warp.deform(f_image.to("cpu"), T, c)
            if irep < 0:
                plt.imshow(t2im(UNORMALIZE(g_image)));
                plt.title(f"T={T}");
                plt.show();
            g_image = g_image.to(dtype=dtype, device=device)
            g = g_image.permute(1, 2, 0).flatten(0, 1)
            for il, lambda_out in enumerate(lambda_out_tab):
                D_fg, _ = estimator(f_coords, f, mask, f_coords, g, lambda_out)
                acc.append(D_fg)
                l2_ = torch.norm(f - g).cpu().item()
                l2.append(l2_)
                T_tab_.append(T)
acc = np.asarray(acc)
l2 = np.asarray(l2)
T = np.asarray(T_tab_)
palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c',
           '#dede00']  # color-blind palette
colors = palette[:len(T_tab)]
markers = ['x', '+', '^', '.', 'v', '<']
for t, marker, color in zip(T_tab, markers, colors):
    index = T == t
    plt.scatter(l2[index], acc[index], marker=marker, c=color);
plt.xscale("log")
plt.yscale("log")
"""
(from other experiments)
Diff: 3.35e-02 < Diff < 9.15e-01
L2: 3.12e+02 < Diff < 5.99e+02
"""
plt.fill_betweenx(np.linspace(acc.min(), 9.15e-1, 1000), 3.12e2, 5.99e2, color=palette[-1], alpha=0.5)
plt.fill_between(np.linspace(l2.min(), l2.max(), 1000), 3.35e-2, 9.15e-1, color=palette[-2], alpha=0.5)
SAVE = True
if not SAVE:
    plt.show()
else:
    plt.tight_layout()
    plt.savefig("icml_figures/warping_random_raw_6.svg")
