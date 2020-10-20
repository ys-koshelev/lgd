from typing import Callable

import torch as th
import torch.nn.functional as F

from .base import LinearDegradationBase


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
        Method which performs a transposed convolution K^T x on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1], which should be degraded
        :return:batch of degraded images [B, C2, H2, W2]
        """
        assert latent_images.shape[0] == self.kernels.shape[0], f'Number of images in batch is not equal to number ' \
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
        ret = F.conv2d(images.view((images.shape[0], *images.shape[-3:])).transpose(1, 0),
                       th.flip(kernels.view((kernels.shape[0], *kernels.shape[-3:])), dims=(-1, -2)),
                       groups=kernels.shape[0]).transpose(1, 0)
        return ret

    @staticmethod
    def _valid_convolve_transposed(images: th.Tensor, kernels: th.Tensor) -> th.Tensor:
        """
        Method, which performs a transposed valid convolution of batch of images with batch of kernels

        :param images: batch of images of shape [B, C, H, W]
        :param kernels: batch of kernels of shape [B, 1, h, w]
        :return: transposed convolved images of shape [B, C, H-h, W-w]
        """
        ret = F.conv_transpose2d(images.view((images.shape[0], *images.shape[-3:])).transpose(1, 0),
                                 th.flip(kernels.view((kernels.shape[0], *kernels.shape[-3:])), dims=(-1, -2)),
                                 groups=kernels.shape[0]).transpose(1, 0)
        return ret


class DownscaleDegradation(BlurDegradation):
    """
    This is a class for downscale degradation of the form y = D(k * x) + n = D K x + n,
    where D - decimation operator, k - blur kernel, K - convolution matrix, n - i.i.d. Gaussian noise
    """
    def __init__(self, scale_factor: int, downscale_kernels: th.Tensor, likelihood_loss: Callable = F.mse_loss) -> None:
        """
        Initializing kernels and scale factor needed to perfrom degradation
        """
        super().__init__(downscale_kernels, likelihood_loss)
        self.scale_factor = scale_factor

    def linear_transform(self, latent_images: th.Tensor) -> th.Tensor:
        """
        Method which performs a linear downscale degradation D(k * x) = D K x on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1], which should be degraded
        :return: batch of degraded images [B, C2, H2, W2]
        """
        assert latent_images.shape[0] == self.kernels.shape[0], f'Number of images in batch is not equal to number ' \
                                                                f'of kernels. Given {latent_images.shape[0]} images ' \
                                                                f'and {self.kernels.shape[0]} kernels.'
        if latent_images.device != self.kernels.device:
            self.kernels = self.kernels.to(latent_images.device)
        return self._decimate(self._valid_convolve(latent_images, self.kernels))

    def linear_transform_transposed(self, latent_images: th.Tensor) -> th.Tensor:
        """
        Method which performs a transposed downscaling K^T D^T x on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1], which should be degraded
        :return:batch of degraded images [B, C2, H2, W2]
        """
        assert latent_images.shape[0] == self.kernels.shape[0], f'Number of images in batch is not equal to number ' \
                                                                f'of kernels. Given {latent_images.shape[0]} images ' \
                                                                f'and {self.kernels.shape[0]} kernels.'
        if latent_images.device != self.kernels.device:
            self.kernels = self.kernels.to(latent_images.device)
        return self._valid_convolve_transposed(self._decimate_transposed(latent_images), self.kernels)

    def _decimate(self, images: th.Tensor) -> th.Tensor:
        """
        Class that performs downscaling of input image by decimation operation.
        Example of such operation for 1/2 downscaling is given below:
        ...|+x| |+x| |+x| |+x|...
        ...|xx| |xx| |xx| |xx|...
        -------------------------
        ...|+x| |+x| |+x| |+x|...
        ...|xx| |xx| |xx| |xx|...
        Here  '+' - pixels, from which the downscaled image will be constructed.
        We can think about this operation as firstly dividing image by superpixels and secondly selecting one pixel per
        each superpixel.

        :param images: batch of input images with shape [B, C, H, W] to downscale
        :return: batch of downscaled images with shape [B, C, H//self.scale_factor, W//self.scale_factor]
        """
        assert images.shape[-1] % self.scale_factor == 0 and images.shape[-2] % self.scale_factor == 0
        coordinate = self._get_pixel_coords_for_decimation()
        fold = images.unfold(-2, self.scale_factor, self.scale_factor).unfold(-2, self.scale_factor, self.scale_factor)
        return fold[..., coordinate, coordinate]

    def _decimate_transposed(self, images: th.Tensor) -> th.Tensor:
        """
        Class that performs upscaling of input image by transposed decimation operation.
        Example of such operation for 2x upscaling is given below:
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        -------------------------
        ...|+0| |+0| |+0| |+0|...
        ...|00| |00| |00| |00|...
        Here  '+' - pixels from input image, which are padded with zeros to construct image of higher resolution.

        :param images: batch of input images with shape [B, C, H, W] to upscale
        :return: batch of upscaled images with shape [B, C, H*self.scale_factor, W*self.scale_factor]
        """
        coordinate = self._get_pixel_coords_for_decimation()
        hw = [i * self.scale_factor for i in images.shape[2:]]
        upscaled = th.zeros(*images.shape[:2], *hw, dtype=images.dtype, device=images.device)
        upscaled = upscaled.unfold(-2, self.scale_factor, self.scale_factor).unfold(-2, self.scale_factor,
                                                                                    self.scale_factor)
        upscaled[:, :, :, :, coordinate, coordinate] = images
        upscaled = upscaled.permute(0, 1, 4, 5, 2, 3).flatten(start_dim=-2).flatten(start_dim=1, end_dim=-2)
        upscaled = F.fold(upscaled, hw, self.scale_factor, stride=self.scale_factor)
        return upscaled

    def _get_pixel_coords_for_decimation(self) -> int:
        """
        Auxiliary method to select coordinates of pixel, which will be used in decimation.
        If scale factor is odd, then central pixel is selected. Otherwise - first at to the top-left from the central.

        :return: coordinate (same for both horizontal and vertical axes)
        """
        return self.scale_factor//2 + self.scale_factor%2 - 1

    def init_latent_images(self, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method is used to init latent images from degraded ones for the first restoration step.
        For linear downscale degradation nearest neighbours upsampling suits better for initialization.
        This is similar to transposed decimation, but zeros are filled with neighbouring pixels intensity values.

        :param degraded_images: batch of images of shape [B, C1, H1, W1] to create latent ones
        :return: initialized latent images of shape [B, C2, H2, W2]
        """
        coordinate = self._get_pixel_coords_for_decimation()
        hw = [i * self.scale_factor for i in degraded_images.shape[2:]]
        upscaled = th.zeros(*degraded_images.shape[:2], *hw, dtype=degraded_images.dtype, device=degraded_images.device)
        upscaled = upscaled.unfold(-2, self.scale_factor, self.scale_factor).unfold(-2, self.scale_factor,
                                                                                    self.scale_factor)
        upscaled[:, :, :, :, coordinate, coordinate] = degraded_images.unsqueeze(0).unsqueeze(0)
        upscaled = upscaled.permute(0, 1, 4, 5, 2, 3).flatten(start_dim=-2).flatten(start_dim=1, end_dim=-2)
        upscaled = F.fold(upscaled, hw, self.scale_factor, stride=self.scale_factor)
        return upscaled
