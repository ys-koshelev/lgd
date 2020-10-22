import numpy as np

from forked.SPADE.data.ade20k_dataset import ADE20KDataset
from .kernels import GaussianKernelSampler, ShakeKernelSampler


class NoiseDataset(ADE20KDataset):
    """
    Wrapper, which adds noise stds to dataset
    """
    def __init__(self, min_std: float, max_std: float, dataset_root: str) -> None:
        """
        :param min_std: lower bound value of noise standard deviation to use in degradation
        :param max_std: upper bound value of noise standard deviation to use in degradation
        :param dataset_root: path to dataset root folder
        """
        self.min_std = min_std
        self.max_std = max_std
        self.initialize({'dataroot': dataset_root})

    def __getitem__(self, item):
        data = super().__getitem__(item)
        noise_std = self.min_std + np.random.rand()*(self.max_std - self.min_std)
        data.update({'noise_std': noise_std})
        return data


class DownscalingDataset(NoiseDataset):
    """
    Wrapper, which adds downscale kernels to dataset
    """
    def __init__(self, min_std: float, max_std: float, dataset_root: str, scale_factor: int, kernel_size: int = 13,
                 ) -> None:
        """
        :param min_std: lower bound value of noise standard deviation to use in degradation
        :param max_std: upper bound value of noise standard deviation to use in degradation
        :param dataset_root: path to dataset root folder
        :param scale_factor: scale factor of super-resolution problem
        :param kernel_size: size of canvas to be used for sampling
        """
        super().__init__(min_std, max_std, dataset_root)
        self.kernels_sampler = GaussianKernelSampler(kernel_size, scale_factor)

    def __getitem__(self, item):
        data = super().__getitem__(item)
        kernel = self.kernels_sampler.sample_kernel()[None, :, :]
        data.update({'kernel': kernel})
        return data


class ShakeBlurDataset(NoiseDataset):
    """
    Wrapper, which adds shake blur kernels to dataset
    """
    def __init__(self, min_std: float, max_std: float, dataset_root: str, kernel_size: int = 21,
                 ) -> None:
        """
        :param min_std: lower bound value of noise standard deviation to use in degradation
        :param max_std: upper bound value of noise standard deviation to use in degradation
        :param dataset_root: path to dataset root folder
        :param kernel_size: size of canvas to be used for sampling
        """
        super().__init__(min_std, max_std, dataset_root)
        self.kernels_sampler = ShakeKernelSampler(kernel_size)

    def __getitem__(self, item):
        data = super().__getitem__(item)
        kernel = self.kernels_sampler.sample_kernel()[None, :, :]
        data.update({'kernel': kernel})
        return data
