from __future__ import print_function
import os
import errno
import struct
import os.path
import hashlib

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision import transforms

NORB_ROOT = '/mnt/fs5/wumike/datasets/norb'
NUMPY_ROOT = os.path.join(NORB_ROOT, 'numpy')

ALL_ELEVATION = np.arange(9)
DEFAULT_ELEVATION = ALL_ELEVATION[0::2]
UNSEEN_ELEVATION = ALL_ELEVATION[1::2]
DEFAULT_ELEVATION_SPARSE = np.array([1,4,7])
UNSEEN_ELEVATION_SPARSE = np.array([0,2,3,5,6,8])
DEFAULT_ELEVATION_DISJOINT = ALL_ELEVATION[:5]
UNSEEN_ELEVATION_DISJOINT = ALL_ELEVATION[5:]
DEFAULT_ELEVATION_DICT = {
    'standard': DEFAULT_ELEVATION,
    'sparse': DEFAULT_ELEVATION_SPARSE,
    'disjoint': DEFAULT_ELEVATION_DISJOINT
}
UNSEEN_ELEVATION_DICT = {
    'standard': UNSEEN_ELEVATION,
    'sparse': UNSEEN_ELEVATION_SPARSE,
    'disjoint': UNSEEN_ELEVATION_DISJOINT
}

ALL_AZIMUTH = np.arange(18)
DEFAULT_AZIMUTH = ALL_AZIMUTH[0::2]
UNSEEN_AZIMUTH = ALL_AZIMUTH[1::2]
DEFAULT_AZIMUTH_SPARSE = np.array([0,4,8,12,16])
UNSEEN_AZIMUTH_SPARSE = np.array([1,2,3,5,6,7,9,10,11,13,14,15,17])
DEFAULT_AZIMUTH_DISJOINT = ALL_AZIMUTH[:9]
UNSEEN_AZIMUTH_DISJOINT = ALL_AZIMUTH[9:]
DEFAULT_AZIMUTH_DICT = {
    'standard': DEFAULT_AZIMUTH,
    'sparse': DEFAULT_AZIMUTH_SPARSE,
    'disjoint': DEFAULT_AZIMUTH_DISJOINT
}
UNSEEN_AZIMUTH_DICT = {
    'standard': UNSEEN_AZIMUTH,
    'sparse': UNSEEN_AZIMUTH_SPARSE,
    'disjoint': UNSEEN_AZIMUTH_DISJOINT
}

ALL_LIGHTING = np.arange(6)
DEFAULT_LIGHTING = ALL_LIGHTING[0::2]
UNSEEN_LIGHTING = ALL_LIGHTING[1::2]
DEFAULT_LIGHTING_SPARSE = np.array([1,4])
UNSEEN_LIGHTING_SPARSE = np.array([0,2,3,5])
DEFAULT_LIGHTING_DISJOINT = ALL_LIGHTING[:3]
UNSEEN_LIGHTING_DISJOINT = ALL_LIGHTING[3:]
DEFAULT_LIGHTING_DICT = {
    'standard': DEFAULT_LIGHTING,
    'sparse': DEFAULT_LIGHTING_SPARSE,
    'disjoint': DEFAULT_LIGHTING_DISJOINT
}
UNSEEN_LIGHTING_DICT = {
    'standard': UNSEEN_LIGHTING,
    'sparse': UNSEEN_LIGHTING_SPARSE,
    'disjoint': UNSEEN_LIGHTING_DISJOINT
}


def load_norb(data_dir, image_size=32, train=True):
    image_transforms = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    image_transforms = transforms.Compose(image_transforms)
    dset = SmallNORB(data_dir, train=train, transform=image_transforms, download=True)
    return dset


def load_norb_numpy(index, concept='elevation', image_size=32, train=True):
    if concept == 'elevation':
        dset = SmallNORB_Many_Elevation(index, train=train)
    elif concept == 'azimuth':
        dset = SmallNORB_Many_Azimuth(index, train=train)
    elif concept == 'lighting':
        dset = SmallNORB_Many_Lighting(index, train=train)
    else:
        raise Exception('split {} not recognized'.format(split))
    return dset


class SmallNORB_Many_Elevation(data.Dataset):
    def __init__(self, index, train=True):
        split = 'train' if train else 'test'
        dset = np.load(os.path.join(NUMPY_ROOT, '{}_elevation_dataset_{}.npz'.format(split, index)))
        imgs, infos, targets = dset['imgs'], dset['infos'], dset['targets']
        self.imgs, self.infos, self.targets = imgs, infos, targets
        self.n = len(self.imgs)
        self.index = index

    def __getitem__(self, index):
        img = self.imgs[index]
        target = self.targets[index]
        return img, target

    def  __len__(self):
        return self.n


class SmallNORB_Many_Azimuth(data.Dataset):
    def __init__(self, index, train=True):
        split = 'train' if train else 'test'
        dset = np.load(os.path.join(NUMPY_ROOT, '{}_azimuth_dataset_{}.npz'.format(split, index)))
        imgs, infos, targets = dset['imgs'], dset['infos'], dset['targets']
        self.imgs, self.infos, self.targets = imgs, infos, targets
        self.n = len(self.imgs)
        self.index = index

    def __getitem__(self, index):
        img = self.imgs[index]
        target = self.targets[index]
        return img, target

    def  __len__(self):
        return self.n


class SmallNORB_Many_Lighting(data.Dataset):
    def __init__(self, index, train=True):
        split = 'train' if train else 'test'
        dset = np.load(os.path.join(NUMPY_ROOT, '{}_lighting_dataset_{}.npz'.format(split, index)))
        imgs, infos, targets = dset['imgs'], dset['infos'], dset['targets']
        self.imgs, self.infos, self.targets = imgs, infos, targets
        self.n = len(self.imgs)
        self.index = index

    def __getitem__(self, index):
        img = self.imgs[index]
        target = self.targets[index]
        return img, target

    def  __len__(self):
        return self.n


class SmallNORB(data.Dataset):
    """`MNIST <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small//>`_ Dataset.

    Args:
        root (string): Root directory of dataset where processed folder and
            and  raw folder exist.
        train (bool, optional): If True, creates dataset from the training files,
            otherwise from the test files.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If the dataset is already processed, it is not processed
            and downloaded again. If dataset is only already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        info_transform (callable, optional): A function/transform that takes in the
            info and transforms it.
        mode (string, optional): Denotes how the images in the data files are returned. Possible values:
            - all (default): both left and right are included separately.
            - stereo: left and right images are included as corresponding pairs.
            - left: only the left images are included.
            - right: only the right images are included.
    """

    dataset_root = "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    data_files = {
        'train': {
            'dat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
                "md5_gz": "66054832f9accfe74a0f4c36a75bc0a2",
                "md5": "8138a0902307b32dfa0025a36dfa45ec"
            },
            'info': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat',
                "md5_gz": "51dee1210a742582ff607dfd94e332e3",
                "md5": "19faee774120001fc7e17980d6960451"
            },
            'cat': {
                "name": 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
                "md5_gz": "23c8b86101fbf0904a000b43d3ed2fd9",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
        'test': {
            'dat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
                "md5_gz": "e4ad715691ed5a3a5f138751a4ceb071",
                "md5": "e9920b7f7b2869a8f1a12e945b2c166c"
            },
            'info': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat',
                "md5_gz": "a9454f3864d7fd4bb3ea7fc3eb84924e",
                "md5": "7c5b871cc69dcadec1bf6a18141f5edc"
            },
            'cat': {
                "name": 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
                "md5_gz": "5aa791cd7e6016cf957ce9bdb93b8603",
                "md5": "fd5120d3f770ad57ebe620eb61a0b633"
            },
        },
    }

    raw_folder = 'raw'
    processed_folder = 'processed'
    train_image_file = 'train_img'
    train_label_file = 'train_label'
    train_info_file = 'train_info'
    test_image_file = 'test_img'
    test_label_file = 'test_label'
    test_info_file = 'test_info'
    extension = '.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, info_transform=None, download=False,
                 mode="all"):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.info_transform = info_transform
        self.train = train  # training set or test set
        self.mode = mode

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # load test or train set
        image_file = self.train_image_file if self.train else self.test_image_file
        label_file = self.train_label_file if self.train else self.test_label_file
        info_file = self.train_info_file if self.train else self.test_info_file

        # load labels
        self.labels = self._load(label_file)

        # t stload info files
        # Each "-info" file stores 24,300 4-dimensional vectors, which contain additional information about the corresponding images:
        # - 1. the instance in the category (0 to 9)
        # - 2. the elevation (0 to 8, which mean cameras are 30, 35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
        # - 3. the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in degrees)
        # - 4. the lighting condition (0 to 5)
        self.infos = self._load(info_file)

        # load right set
        if self.mode == "left":
            self.data = self._load("{}_left".format(image_file))

        # load left set
        elif self.mode == "right":
            self.data = self._load("{image_file}_right".format(image_file))

        elif self.mode == "all" or self.mode == "stereo":
            left_data = self._load("{}_left".format(image_file))
            right_data = self._load("{}_right".format(image_file))

            # load stereo
            if self.mode == "stereo":
                self.data = torch.stack((left_data, right_data), dim=1)

            # load all
            else:
                self.data = torch.cat((left_data, right_data), dim=0)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            mode ``all'', ``left'', ``right'':
                tuple: (image, info, target)
            mode ``stereo'':
                tuple: (image left, image right, info, target)
        """
        target = self.labels[index % 24300] if self.mode is "all" else self.labels[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        info = self.infos[index % 24300] if self.mode is "all" else self.infos[index]
        if self.info_transform is not None:
            info = self.info_transform(info)

        if self.mode == "stereo":
            img_left = self._transform(self.data[index, 0])
            img_right = self._transform(self.data[index, 1])
            return img_left, img_right, info, target

        img = self._transform(self.data[index])
        return img, info, target

    def __len__(self):
        return len(self.data)

    def _transform(self, img):
        # doing this so that it is consistent with all other data sets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def _load(self, file_name):
        return torch.load(os.path.join(self.root, self.processed_folder, "{}{}".format(file_name, self.extension)))

    def _save(self, file, file_name):
        with open(os.path.join(self.root, self.processed_folder, "{}{}".format(file_name, self.extension)), 'wb') as f:
            torch.save(file, f)

    def _check_exists(self):
        """ Check if processed files exists."""
        files = (
            "{}_left".format(self.train_image_file),
            "{}_right".format(self.train_image_file),
            "{}_left".format(self.test_image_file),
            "{}_right".format(self.test_image_file),
            self.test_label_file,
            self.train_label_file
        )
        fpaths = [os.path.exists(os.path.join(self.root, self.processed_folder, "{}{}".format(f, self.extension))) for f in files]
        return False not in fpaths

    def _flat_data_files(self):
        return [j for i in self.data_files.values() for j in list(i.values())]

    def _check_integrity(self):
        """Check if unpacked files have correct md5 sum."""
        root = self.root
        for file_dict in self._flat_data_files():
            filename = file_dict["name"]
            md5 = file_dict["md5"]
            fpath = os.path.join(root, self.raw_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        """Download the SmallNORB data if it doesn't exist in processed_folder already."""
        import gzip

        if self._check_exists():
            return

        # check if already extracted and verified
        if self._check_integrity():
            print('Files already downloaded and verified')
        else:
            # download and extract
            for file_dict in self._flat_data_files():
                url = self.dataset_root + file_dict["name"] + '.gz'
                filename = file_dict["name"]
                gz_filename = filename + '.gz'
                md5 = file_dict["md5_gz"]
                fpath = os.path.join(self.root, self.raw_folder, filename)
                gz_fpath = fpath + '.gz'

                # download if compressed file not exists and verified
                download_url(url, os.path.join(self.root, self.raw_folder), gz_filename, md5)

                print('# Extracting data {}\n'.format(filename))

                with open(fpath, 'wb') as out_f, \
                        gzip.GzipFile(gz_fpath) as zip_f:
                    out_f.write(zip_f.read())

                os.unlink(gz_fpath)

        # process and save as torch files
        print('Processing...')

        # create processed folder
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # read train files
        left_train_img, right_train_img = self._read_image_file(self.data_files["train"]["dat"]["name"])
        train_info = self._read_info_file(self.data_files["train"]["info"]["name"])
        train_label = self._read_label_file(self.data_files["train"]["cat"]["name"])

        # read test files
        left_test_img, right_test_img = self._read_image_file(self.data_files["test"]["dat"]["name"])
        test_info = self._read_info_file(self.data_files["test"]["info"]["name"])
        test_label = self._read_label_file(self.data_files["test"]["cat"]["name"])

        # save training files
        self._save(left_train_img, "{}_left".format(self.train_image_file))
        self._save(right_train_img, "{}_right".format(self.train_image_file))
        self._save(train_label, "{}".format(self.train_label_file))
        self._save(train_info, "{}".format(self.train_info_file))

        # save test files
        self._save(left_test_img, "{}_left".format(self.test_image_file))
        self._save(right_test_img, "{}_right".format(self.test_image_file))
        self._save(test_label, "{}".format(self.test_label_file))
        self._save(test_info, "{}".format(self.test_info_file))

        print('Done!')

    @staticmethod
    def _parse_header(file_pointer):
        # Read magic number and ignore
        struct.unpack('<BBBB', file_pointer.read(4))  # '<' is little endian)

        # Read dimensions
        dimensions = []
        num_dims, = struct.unpack('<i', file_pointer.read(4))  # '<' is little endian)
        for _ in range(num_dims):
            dimensions.extend(struct.unpack('<i', file_pointer.read(4)))

        return dimensions

    def _read_image_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            dimensions = self._parse_header(f)
            assert dimensions == [24300, 2, 96, 96]
            num_samples, _, height, width = dimensions

            left_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)
            right_samples = np.zeros(shape=(num_samples, height, width), dtype=np.uint8)

            for i in range(num_samples):

                # left and right images stored in pairs, left first
                left_samples[i, :, :] = self._read_image(f, height, width)
                right_samples[i, :, :] = self._read_image(f, height, width)

        return torch.ByteTensor(left_samples), torch.ByteTensor(right_samples)

    @staticmethod
    def _read_image(file_pointer, height, width):
        """Read raw image data and restore shape as appropriate. """
        image = struct.unpack('<' + height * width * 'B', file_pointer.read(height * width))
        image = np.uint8(np.reshape(image, newshape=(height, width)))
        return image

    def _read_label_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:
            num_samples = self._parse_header(f)
            assert num_samples == [24300]
            num_samples = num_samples[0]

            struct.unpack('<BBBB', f.read(4))  # ignore this integer
            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            labels = np.zeros(shape=num_samples, dtype=np.int32)
            for i in range(num_samples):
                category, = struct.unpack('<i', f.read(4))
                labels[i] = category
            return torch.LongTensor(labels)

    def _read_info_file(self, file_name):
        fpath = os.path.join(self.root, self.raw_folder, file_name)
        with open(fpath, mode='rb') as f:

            dimensions = self._parse_header(f)
            assert dimensions == [24300, 4]
            num_samples, num_info = dimensions

            struct.unpack('<BBBB', f.read(4))  # ignore this integer

            infos = np.zeros(shape=(num_samples, num_info), dtype=np.int32)

            for r in range(num_samples):
                for c in range(num_info):
                    info, = struct.unpack('<i', f.read(4))
                    infos[r, c] = info

        return torch.LongTensor(infos)


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)

