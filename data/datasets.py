import numpy as np

from forked.SPADE.data.ade20k_dataset import ADE20KDataset
from .kernels import GaussianKernelSampler, ShakeKernelSampler
from argparse import Namespace

class NoiseDataset(ADE20KDataset):
    """
    Wrapper, which adds noise stds to dataset
    """
    def __init__(self, min_std: float, max_std: float, dataset_root: str, train_phase: str = 'train',
                 max_dataset_length: int = 1000, out_images_size: int = 256) -> None:
        """
        :param min_std: lower bound value of noise standard deviation to use in degradation
        :param max_std: upper bound value of noise standard deviation to use in degradation
        :param dataset_root: path to dataset root folder
        :param train_phase: should be either train, val or test
        :param max_dataset_length: maximum length of dataset
        :param out_images_size: size of images to be outputed by this dataset
        """
        self.min_std = min_std
        self.max_std = max_std
        opts = self.get_options(dataset_root, train_phase, max_dataset_length, out_images_size)
        self.initialize(opts)

    def __getitem__(self, item):
        data = super().__getitem__(item)
        noise_std = self.min_std + np.random.rand()*(self.max_std - self.min_std)
        data.update({'noise_std': noise_std})
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
