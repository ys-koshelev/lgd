import torch as th
from .base import ParametricFunctionBase


class IdentityProjection(ParametricFunctionBase):
    """
    This class implements identity projection: project(x) = x
    """
    def call(self, images: th.Tensor) -> th.Tensor:
        """
        Return input images without doing anything

        :param images: batch of input images of shape [B, C, H, W] to be projected
        :return: projected images of shape [B, C, H, W]
        """
        return images


class ClampProjection(ParametricFunctionBase):
    """
    Implements projection by clamping values of input tensor
    """
    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        """
        Initialize parameters

        :param min_value: all input values below this quantity will be clamped from below
        :param max_value: all input values greater this quantity will be clamped from above
        """
        self.min_value = min_value
        self.max_value = max_value

    def call(self, images: th.Tensor) -> th.Tensor:
        """
        Implements identity projection - return input images without doing anything

        :param images: batch of input images of shape [B, C, H, W] to be projected
        :return: projected images of shape [B, C, H, W]
        """
        return th.clamp(images, self.min_value, self.max_value)
