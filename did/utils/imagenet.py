# from __future__ import print_function
import os
import os.path
import sys

import numpy as np
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch

IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

PATH_TO_IMAGENET_TRAIN = ""  #None
PATH_TO_IMAGENET_TEST = ""  #None

if PATH_TO_IMAGENET_TRAIN is None:
    raise FileNotFoundError("Replace Imagenette train directory in imagenet.py.")


class Imagenette(ImageFolder):
    """
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(Imagenette, self).__init__(root, transform=transform,
                                         target_transform=target_transform)


def load_imagenet(traindir, valdir, resize_size, batch_size, num_workers, n_classes=None, shuffle=False):
    trainset = ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(resize_size),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    testset = ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(resize_size),
            transforms.ToTensor(),
            IMAGENET_NORMALIZE,
        ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
    if n_classes is None:
        n_classes = 1000
    return trainset, testset, trainloader, testloader, n_classes
