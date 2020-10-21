import torch as th
from .base import ParametricFunctionBase


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
