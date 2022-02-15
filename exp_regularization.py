import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from kornia import tensor_to_image as t2im

from did.utils import imagenet
import warping.warp as warp
from did import kernels, masking
from did import nystrom
from did.dissimilarity import NormalizedDIDEstimator
from did.utils.imagenet import IMAGENET_NORMALIZE, load_imagenet

SEED = 15
torch.manual_seed(SEED)
np.random.seed(SEED)

image_size = 150
H, W = image_size, image_size
rect_size = 50
x = 10
y = 70
bbox = (x, y, x + rect_size, y + rect_size)
bbox2 = (x + 20, y + 20, x + 20 + rect_size, y + 20 + rect_size)
fill = (0, 0, 0)

path_train = imagenet.PATH_TO_IMAGENET_TRAIN
path_test = imagenet.PATH_TO_IMAGENET_TEST
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
im0 = trainset[image_i][0]

T = 1e-3
c = 2
f_image = im0

dtype = torch.float64
device = "cuda:0"
f_image = f_image.to(dtype=dtype, device=device)

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
plt.imshow(t2im(UNORMALIZE(f_image)))  # *mask_image))
plt.title("f")
plt.show()
# plt.imshow(t2im(UNORMALIZE(g_image)))
# plt.title("g")
# plt.show()

in_params = {"sigma": 1 / 6.0}
in_kernel = kernels.Kernel("gaussian", kernels.gaussian_kernel, in_params)
out_params = {"a": 5.0}
out_kernel = kernels.Kernel("abel", kernels.abel_kernel, out_params)

estimator = NormalizedDIDEstimator(in_kernel, out_kernel, in_nystrom, out_nystrom)
estimator = estimator.pre_compute(in_cholesky_reg)

f = f_image.permute(1, 2, 0).flatten(0, 1)

# note that we assume that f and g have the same coords here (can be changed)
n_rep = 50
lambda_out_tab = [1e-3, 1e-2, 1e-1, 1]  # np.logspace(-10, 0, 3)
T_tab = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 10]
acc = np.zeros((len(T_tab), len(lambda_out_tab), n_rep))
l2 = np.zeros((len(T_tab), len(lambda_out_tab), n_rep))
for it, T in enumerate(T_tab):
    for irep in range(n_rep):
        g_image = im0 if T == 0 else warp.deform(f_image.to("cpu"), T, c)
        if irep == 0:
            plt.imshow(t2im(UNORMALIZE(g_image)));
            plt.show();
        g_image = g_image.to(dtype=dtype, device=device)
        g = g_image.permute(1, 2, 0).flatten(0, 1)
        for il, lambda_out in enumerate(lambda_out_tab):
            D_fg, _ = estimator(f_coords, f, mask, f_coords, g, lambda_out)
            acc[it, il, irep] = D_fg
            l2_ = torch.norm(f - g).cpu().item()
            l2[it, il, irep] = l2_

cm = plt.cm
palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
colors = palette[:len(lambda_out_tab)]

plt.clf()
for il, (c, lambda_out) in enumerate(zip(colors, lambda_out_tab)):
    plt.plot(T_tab, acc[:, il, :].mean(-1), c=c,
             label=rf"$\lambda_{{out}}$ = 1e{int(np.log10(lambda_out))}"
             )
    plt.scatter(T_tab, acc[:, il, :], color="k", alpha=0.2)
    plt.fill_between(T_tab, acc[:, il, :].mean(-1) - acc[:, il, :].std(-1),
                     acc[:, il, :].mean(-1) + acc[:, il, :].std(-1), color="k", alpha=0.2)
plt.xscale("log")
plt.yscale("log")
plt.legend()
SAVE = False
if not SAVE:
    plt.show()
else:
    plt.savefig("icml_figures/regularization.svg")
