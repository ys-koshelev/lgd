from argparse import Namespace
import abc
import numpy as np
import torch as th
from torch.utils.data import Dataset
from typing import Dict, Any
from forked.SPADE.data.ade20k_dataset import ADE20KDataset
from .kernels import GaussianKernelSampler, ShakeKernelSampler
from glob import glob
import os.path as osp
import cv2
import random


class BSD500ImagesDataset(ADE20KDataset):
    """
    Dataset, which loads BSD500 images
    """
    def __init__(self, dataset_root: str, train_phase: str = 'train', max_dataset_length: int = 1000,
                 out_images_size: int = 256, grayscale_output: bool = False) -> None:
        """
        :param dataset_root: path to dataset root folder
        :param train_phase: should be either train, val or test
        :param max_dataset_length: maximum length of dataset
        :param out_images_size: size of images to be outputed by this dataset
        :param grayscale_output: if True, image is converted to return grayscale
        """
        assert train_phase in ('train', 'val', 'test')
        self.images_paths_list = glob(osp.join(dataset_root, train_phase, '*.jpg'))
        self.grayscale_output = grayscale_output
        self.dataset_length = min(len(self.images_paths_list), max_dataset_length)
        self.crop_size = out_images_size

    def __getitem__(self, item: int) -> Dict[str, Any]:
        image = random.choice(self.images_paths_list)
        image = cv2.imread(image)
        if self.grayscale_output:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.random_crop(image).astype(np.float32)/255
        # image = np.transpose(image, axes=(2, 0, 1))
        return {'image': image}

    def __len__(self) -> int:
        return self.dataset_length

    def random_crop(self, image: np.array) -> np.array:
        h, w = image.shape[:2]
        h_crop = np.random.randint(0, h - self.crop_size)
        w_crop = np.random.randint(0, w - self.crop_size)
        return image[h_crop:h_crop + self.crop_size, w_crop:w_crop + self.crop_size, :]


class ADE20KImageDataset(ADE20KDataset):
    """
    Wrapper, which adds noise stds to dataset
    """
    def __init__(self, dataset_root: str, train_phase: str = 'train', max_dataset_length: int = 1000,
                 out_images_size: int = 256, grayscale_output: bool = False) -> None:
        """
        :param dataset_root: path to dataset root folder
        :param train_phase: should be either train, val or test
        :param max_dataset_length: maximum length of dataset
        :param out_images_size: size of images to be outputed by this dataset
        :param grayscale_output: if True, image is converted to return grayscale
        """
        opts = self.get_options(dataset_root, train_phase, max_dataset_length, out_images_size)
        self.initialize(opts)
        self.grayscale_output = grayscale_output

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = super().__getitem__(item)
        if self.grayscale_output:
            data['image'] = self.color2grayscale(data['image'])
        data['image'] = (data['image'] + 1)/2
        data['image'] = data['image'].permute(1,2,0).numpy()
        data['label'] = data['label'][0].numpy()
        data['label'][data['label'] == 150.] = 149.
        del data['instance']
        del data['path']
        return data

    def get_options(self, root_dir: str, phase: str, max_size: int, crop_size: int) -> Namespace:
        options = Namespace()
        options.dataroot = root_dir
        options.phase = phase
        options.no_instance = True
        options.max_dataset_size = max_size
        options.no_pairing_check = True
        options.preprocess_mode = 'crop'
        options.crop_size = crop_size
        if phase == 'train':
            is_train = True
        else:
            is_train = False
        options.isTrain = is_train
        options.no_flip = False
        options.label_nc = 150
        return options

    @staticmethod
    def color2grayscale(image: th.Tensor) -> th.Tensor:
        """
        Cenerting color image to grayscale

        :param image: input image of shape [C, H, W], given in RGB channels order
        :return: grayscale image of shape [1, H, W]
        """
        assert image.shape[0] == 3
        r, g, b = image
        gray = 0.299*r + 0.587*r + 0.114*b
        return gray.unsqueeze(0)


class DatasetWrapperBase(Dataset):
    """
    Base class for implementing a dataset wrapper, which adds or changes some data to existing dataset
    """
    dataset: Dataset

    @abc.abstractmethod
    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method returns additional changes to data, which should be added to base dataset output

        :param idx: element index
        :param data_dict: dict with data, coming from base dataset, which should be augmented
        :return: dict with augmented data
        """
        pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.dataset[idx]
        data = self.get_additional_data(idx, data)
        return data

    def __len__(self):
        return len(self.dataset)


class NoiseDatasetWrapper(DatasetWrapperBase):
    """
    Wrapper, which adds noise stds to dataset
    """
    def __init__(self, dataset: Dataset, min_std: float, max_std: float) -> None:
        """
        :param dataset: dataset to wrap around the noise sampling
        :param min_std: lower bound value of noise standard deviation to use in degradation
        :param max_std: upper bound value of noise standard deviation to use in degradation
        """
        self.min_std = min_std
        self.max_std = max_std
        self.dataset = dataset

    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add noise std value to data dict

        :param idx: element index
        :param data_dict: dict with data, coming from base dataset, which should be augmented
        :return: dict with added noise std value
        """
        noise_std = self.min_std + np.random.rand()*(self.max_std - self.min_std)
        data_dict.update({'noise_std': noise_std})
        return data_dict


class DownscalingDatasetWrapper(DatasetWrapperBase):
    """
    Wrapper, which adds downscale kernels to dataset
    """
    def __init__(self, dataset: Dataset, scale_factor: int, kernel_size: int = 13) -> None:
        """
        :param scale_factor: scale factor of super-resolution problem
        :param kernel_size: size of canvas to be used for sampling
        """
        self.kernels_sampler = GaussianKernelSampler(kernel_size, scale_factor)
        self.dataset = dataset

    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add downscaling kernel to dataset

        :param idx: element index
        :param data_dict: dict with data, coming from base dataset, which should be augmented
        :return: dict with added downscale kernels
        """
        kernel = self.kernels_sampler.sample_kernel()[None, :, :]
        data_dict.update({'kernels': kernel})
        return data_dict


class ShakeBlurDatasetWrapper(DatasetWrapperBase):
    """
    Wrapper, which adds shake blur kernels to dataset
    """
    def __init__(self, dataset: Dataset, kernel_size: int = 21) -> None:
        """
        :param dataset: dataset to wrap around the noise sampling
        :param kernel_size: size of canvas to be used for sampling
        """
        self.kernels_sampler = ShakeKernelSampler(kernel_size)
        self.dataset = dataset

    def augment_data(self, idx: int, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add camera shake blur kernel to dataset

        :param idx: element index
        :param data_dict: dict with data, coming from base dataset, which should be augmented
        :return: dict with added camera shake blur kernel
        """
        kernel = self.kernels_sampler.sample_kernel()[None, :, :]
        data_dict.update({'kernels': kernel})
        return data_dict
