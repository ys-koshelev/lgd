import torch as th
from .base import LinearDegradationBase
from typing import Callable
import torch.nn.functional as F


class NoiseDegradation(LinearDegradationBase):
    """
    This is a class for additive Gaussian noise degradation of the form y = x + n, where n - i.i.d. Gaussian noise
    """
    def linear_transform(self, latent_images: th.Tensor) -> th.Tensor:
        """
        Method which performs a linear degradation Ix = x on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1], which should be degraded
        :return:batch of degraded images [B, C2, H2, W2]
        """
        return latent_images

    def linear_transform_transposed(self, latent_images: th.Tensor) -> th.Tensor:
        """
        Method which performs a linear degradation I^T x = Ix = x on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1], which should be degraded
        :return:batch of degraded images [B, C2, H2, W2]
        """
        return latent_images


class BlurDegradation(LinearDegradationBase):
    """
    This is a class for blur degradation of the form y = k * x + n, where k - blur kernel, n - i.i.d. Gaussian noise
    """
    def __init__(self, blur_kernels: th.Tensor, likelihood_loss: Callable = F.mse_loss) -> None:
        """
        Initializing everything that is needed to perfrom degradation
        """
        super().__init__(likelihood_loss)
        self.kernels = blur_kernels

    def linear_transform(self, latent_images: th.Tensor) -> th.Tensor:
        """
        Method which performs a linear degradation k * x on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1], which should be degraded
        :return:batch of degraded images [B, C2, H2, W2]
        """
        assert latent_images.shape[0] == self.kernels.shape[0], f'Number of images in batch is not equal to nubler ' \
                                                                f'of kernels. Given {latent_images.shape[0]} images ' \
                                                                f'and {self.kernels.shape[0]} kernels.'
        if latent_images.device != self.kernels.device:
            self.kernels = self.kernels.to(latent_images.device)
        return self._valid_convolve(latent_images, self.kernels)

    def linear_transform_transposed(self, latent_images: th.Tensor) -> th.Tensor:
        """
        Method which performs a linear degradation I^T x = Ix = x on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1], which should be degraded
        :return:batch of degraded images [B, C2, H2, W2]
        """
        assert latent_images.shape[0] == self.kernels.shape[0], f'Number of images in batch is not equal to nubler ' \
                                                                f'of kernels. Given {latent_images.shape[0]} images ' \
                                                                f'and {self.kernels.shape[0]} kernels.'
        if latent_images.device != self.kernels.device:
            self.kernels = self.kernels.to(latent_images.device)
        return self._valid_convolve_transposed(latent_images, self.kernels)

    @staticmethod
    def _valid_convolve(images: th.Tensor, kernels: th.Tensor) -> th.Tensor:
        """
        Method, which performs a valid convolution of batch of images with batch of kernels

        :param images: batch of images of shape [B, C, H, W]
        :param kernels: batch of kernels of shape [B, 1, h, w]
        :return: convolved images of shape [B, C, H-h, W-w]
        """
        ret = F.conv2d(images.view((images.shape[0] * images.shape[1], *images.shape[-3:])).transpose(1, 0),
                       th.flip(kernels.view((kernels.shape[0] * kernels.shape[1], *kernels.shape[-3:])), dims=(-1, -2)),
                       groups=kernels.shape[0] * kernels.shape[1]).transpose(1, 0)
        return ret.view(*images.shape[:2], *ret.shape[-3:])

    @staticmethod
    def _valid_convolve_transposed(images: th.Tensor, kernels: th.Tensor) -> th.Tensor:
        """
        Method, which performs a transposed valid convolution of batch of images with batch of kernels

        :param images: batch of images of shape [B, C, H, W]
        :param kernels: batch of kernels of shape [B, 1, h, w]
        :return: transposed convolved images of shape [B, C, H-h, W-w]
        """
        ret = F.conv_transpose2d(
            images.view((images.shape[0] * images.shape[1], *images.shape[-3:])).transpose(1, 0),
            th.flip(kernels.view((kernels.shape[0] * kernels.shape[1], *kernels.shape[-3:])), dims=(-1, -2)),
            groups=kernels.shape[0] * kernels.shape[1]).transpose(1, 0)
        return ret.view(*images.shape[:2], *ret.shape[-3:])
