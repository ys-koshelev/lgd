import abc
from typing import Dict, Any

import numpy as np
from torch.utils.data import Dataset

from .kernels import GaussianKernelSampler, ShakeKernelSampler


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