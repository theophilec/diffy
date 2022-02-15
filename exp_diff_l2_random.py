import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from did.utils import imagenet
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
device = "cuda" if torch.cuda.is_available() else "cpu"

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
n_images = 50
image_ids_f = np.random.randint(0, len(trainset), n_images)
image_ids_g = np.random.randint(0, len(trainset), n_images)

lambda_out_tab = [1e-6]  # , 1e-4, 1e-3, 1e-2] #np.logspace(-10, 0, 3)
T_tab = [0]  # [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
acc = []
l2 = []
T_tab_ = []
for i, image_id_i in enumerate(image_ids_f):
    print(f"{i}")
    for j, image_id_j in enumerate(image_ids_g):
        im0 = trainset[image_id_i][0]
        f_image = im0
        f_image = f_image.to(dtype=dtype, device=device)

        im0 = trainset[image_id_j][0]
        g_image = im0
        g_image = g_image.to(dtype=dtype, device=device)
        if i == 0 and j == 0:
            mask_image = mask.generate_mu_pt(f_image).to(dtype=dtype).to(device)
            mask = mask_image.flatten().unsqueeze(-1).to(device)
        f = f_image.permute(1, 2, 0).flatten(0, 1)
        g = g_image.permute(1, 2, 0).flatten(0, 1)
        D_fg, _ = estimator(f_coords, f, mask, f_coords, g, lambda_out_tab[0])
        acc.append(D_fg)
        l2_ = torch.norm(f - g).cpu().item()
        l2.append(l2_)
        T_tab_.append(T)
acc = np.asarray(acc)
l2 = np.asarray(l2)
T = np.asarray(T_tab_)

plt.scatter(T, l2, c="blue")
plt.scatter(T, acc, c="red")
plt.xscale("log")
plt.yscale("log")
plt.show()

print("done")

beta = 0.10
acc_lower = np.quantile(acc, beta / 2)
acc_higher = np.quantile(acc, 1 - beta / 2)
l2_lower = np.quantile(l2, beta / 2)
l2_higher = np.quantile(l2, 1 - beta / 2)

print(f"Diff: {acc_lower:.2e} < Diff < {acc_higher:.2e}")
print(f"L2: {l2_lower:.2e} < Diff < {l2_higher:.2e}")