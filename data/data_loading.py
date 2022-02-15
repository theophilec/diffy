import numpy as np
import scipy.io
from PIL import Image


def load_perspective(i):
    i = int(i)
    im = Image.open(f"data/perspective/{i}.jpg")
    return im


def float_array_to_pil(array):
    assert array.ndim == 3
    return Image.fromarray(np.uint8(array * 255))


def load_scene(i):
    i = int(i)
    im = Image.open(f"data/scenes/{i}.jpg")
    return im


def load_peppers():
    import scipy.io
    mat = scipy.io.loadmat('data/peppers.mat')['b']
    return mat