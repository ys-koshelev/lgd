import torch as th
from .base import ParametricFunctionBase
import torch.nn.functional as F


class LpNormPowerPrior(ParametricFunctionBase):
    """
    Implements prior in a form of p(x) = ||x||^g_p
    """
    def __init__(self, p: float, power: float = 1.):
        """
        Initializing all needed parameters

        :param p: norm type
        :param power: power to raise the norm value
        """
        self.p = p
        self.power = power

    def call(self, input_tensor: th.Tensor) -> th.Tensor:
        """
        Computes prior value as ||x||^g_p

        :param input_tensor: batch of input images of shape [B, C, H, W]
        :return: batch of prior values of shape [B]
        """
        return th.pow(th.norm(input_tensor.flatten(start_dim=1), p=self.p), self.power)


class ZeroPrior(ParametricFunctionBase):
    """
    Implements prior in a form of p(x) = 0
    """
    def call(self, input_tensor: th.Tensor) -> th.Tensor:
        """
        Always assign prior value to 0

        :param input_tensor: batch of input images of shape [B, C, H, W]
        :return: batch of prior values of shape [B]
        """
        return th.zeros(input_tensor.shape[0], dtype=input_tensor.dtype, device=input_tensor.device)


class TotalVariationPrior(ParametricFunctionBase):
    def __init__(self):
        self.dx = th.FloatTensor([1, -1])[None, None, None, :]
        self.dy = th.FloatTensor([1, -1])[None, None, :, None]

    def call(self, input_tensor: th.Tensor) -> th.Tensor:
        if self.dx.device != input_tensor.device or self.dx.dtype != input_tensor.dtype:
            self.dx = self.dx.to(input_tensor)
            self.dy = self.dy.to(input_tensor)
        dx = self._valid_convolve(input_tensor, self.dx.expand(input_tensor.shape[0], -1, -1, -1))
        dy = self._valid_convolve(input_tensor, self.dy.expand(input_tensor.shape[0], -1, -1, -1))
        return dx.abs().sum() + dy.abs().sum()


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
