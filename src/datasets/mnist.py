import math
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

# ----- ROTATED MNIST  -----

ROTATIONS = np.arange(-180, 180, 20)
DEFAULT_ROTATIONS = ROTATIONS[0::2]
UNSEEN_ROTATIONS  = ROTATIONS[1::2]
DEFAULT_ROTATIONS_SPARSE = np.array([-160, -80, 0, 80, 160])
UNSEEN_ROTATIONS_SPARSE = np.array([-180, -140, -120, -100, -60, -40, -20, 20, 40, 60, 100, 120, 140])
DEFAULT_ROTATIONS_DISJOINT = ROTATIONS[:len(ROTATIONS) // 2 + 1]
UNSEEN_ROTATIONS_DISJOINT = ROTATIONS[len(ROTATIONS) // 2 + 1:]
ALL_ROTATIONS = ROTATIONS
DEFAULT_ROTATIONS_DICT = {
    'standard': DEFAULT_ROTATIONS,
    'sparse': DEFAULT_ROTATIONS_SPARSE,
    'disjoint': DEFAULT_ROTATIONS_DISJOINT
}
UNSEEN_ROTATIONS_DICT = {
    'standard': UNSEEN_ROTATIONS,
    'sparse': UNSEEN_ROTATIONS_SPARSE,
    'disjoint': UNSEEN_ROTATIONS_DISJOINT
}


def load_many_rotated_mnist(data_dir, image_size=32, train=True,
                            rotations=DEFAULT_ROTATIONS):
    """
    Load 10 different MNIST datasets where the image in each dataset
    has a particular rotation.
    """
    return [
        load_rotated_mnist( data_dir, image_size=image_size, 
                            train=train, rotation=rotation)
        for rotation in rotations
    ]


def load_rotated_mnist(data_dir, image_size=32, train=True, rotation=0):
    """
    Load a MNIST dataset where each image has a rotation.
    """
    rotate_image = rotate_transform(rotation)
    image_transforms = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        rotate_image,
        transforms.ToTensor(),
    ]
    image_transforms = transforms.Compose(image_transforms)
    dset = datasets.MNIST(data_dir, train=train, download=True,
                          transform=image_transforms)
    return dset


def rotate_transform(angle):
    def f(img):
        return transforms.functional.rotate(img, angle)
    return f

# ----- SCALED MNIST  -----

SCALES = np.arange(0.5, 2.0, 0.1)
DEFAULT_SCALES = SCALES[0::2]
UNSEEN_SCALES  = SCALES[1::2]
DEFAULT_SCALES_SPARSE = np.array([0.6, 1.0 ,1.4, 1.8])
UNSEEN_SCALES_SPARSE = np.array([0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.9])
DEFAULT_SCALES_DISJOINT = SCALES[:len(SCALES) // 2 + 1]
UNSEEN_SCALES_DISJOINT = SCALES[len(SCALES) // 2 + 1:]
ALL_SCALES = SCALES
DEFAULT_SCALES_DICT = {
    'standard': DEFAULT_SCALES,
    'sparse': DEFAULT_SCALES_SPARSE,
    'disjoint': DEFAULT_SCALES_DISJOINT
}
UNSEEN_SCALES_DICT = {
    'standard': UNSEEN_SCALES,
    'sparse': UNSEEN_SCALES_SPARSE,
    'disjoint': UNSEEN_SCALES_DISJOINT
}


def load_many_scaled_mnist( data_dir, image_size=32, train=True,
                            scales=DEFAULT_SCALES):
    """
    Load 10 different MNIST datasets where the image in each dataset
    has a particular scale.
    """
    return [
        load_scaled_mnist( data_dir, image_size=image_size, 
                            train=train, scale=scale)
        for scale in scales
    ]


def load_scaled_mnist(data_dir, image_size=32, train=True, scale=1):
    """
    Load a MNIST dataset where each image has is scaled by a scale.
    """
    scale_image = scale_transform(scale)
    image_transforms = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        scale_image,
        transforms.ToTensor(),
    ]
    image_transforms = transforms.Compose(image_transforms)
    dset = datasets.MNIST(data_dir, train=train, download=True,
                          transform=image_transforms)
    return dset


def scale_transform(scale):
    def f(img):
        size = img.size
        i, j, h, w = get_crop_params(img, scale, ratio=1)
        return transforms.functional.resized_crop(
            img, i, j, h, w, size, Image.BILINEAR)
    return f


def get_crop_params(img, scale, ratio=1):
    w = img.size[0] * scale
    h = img.size[1] * scale
    i = (img.size[1] - h) // 2
    j = (img.size[0] - w) // 2
    return i, j, h, w

# ----- SHEARED MNIST  -----

SHEARS = np.arange(-180, 180, 20)
DEFAULT_SHEARS = SHEARS[0::2]
UNSEEN_SHEARS  = SHEARS[1::2]
DEFAULT_SHEARS_SPARSE = np.array([-160, -80, 0, 80, 160])
UNSEEN_SHEARS_SPARSE = np.array([-180, -140, -120, -100, -60, -40, -20, 20, 40, 60, 100, 120, 140])
DEFAULT_SHEARS_DISJOINT = SHEARS[:len(SHEARS) // 2 + 1]
UNSEEN_SHEARS_DISJOINT = SHEARS[len(SHEARS) // 2 + 1:]
ALL_SHEARS = SHEARS
DEFAULT_SHEARS_DICT = {
    'standard': DEFAULT_SHEARS,
    'sparse': DEFAULT_SHEARS_SPARSE,
    'disjoint': DEFAULT_SHEARS_DISJOINT
}
UNSEEN_SHEARS_DICT = {
    'standard': UNSEEN_SHEARS,
    'sparse': UNSEEN_SHEARS_SPARSE,
    'disjoint': UNSEEN_SHEARS_DISJOINT
}


def load_many_sheared_mnist(data_dir, image_size=32, train=True,
                            shears=DEFAULT_SHEARS):
    """
    Load 10 different MNIST datasets where the image in each dataset
    has a particular shear.
    """
    return [
        load_sheared_mnist( data_dir, image_size=image_size, 
                            train=train, shear=shear)
        for shear in shears
    ]


def load_sheared_mnist(data_dir, image_size=32, train=True, shear=0):
    """
    Load a MNIST dataset where each image has a rotation.
    """
    shear_image = shear_transform(shear)
    image_transforms = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        shear_image,
        transforms.ToTensor(),
    ]
    image_transforms = transforms.Compose(image_transforms)
    dset = datasets.MNIST(data_dir, train=train, download=True,
                          transform=image_transforms)
    return dset


def shear_transform(shear):
    def f(img):
        return transforms.functional.affine(img, 0, (0, 0), 1, shear)
    return f
