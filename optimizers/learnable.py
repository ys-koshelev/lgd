from .base import OptimizerBase
import torch.nn as nn
import torch as th
from typing import List, Callable
from degradations import DegradationBase
from functions.projections import IdentityProjection


class LGDOptimizer(nn.Module, OptimizerBase):
    """
    Class which implements learnable gradient descent optimizer as a recurrent convolutional neural network.
    """
    backbone: nn.Module
    backprop_through_gradlikelihood: bool

    def __init__(self, degradation: DegradationBase, optimizer_network: nn.Module, num_steps: int,
                 projection_function: Callable = None, backprop_through_gradlikelihood: bool = False) -> None:
        """
        Initialize a learnable gradient descent optimizer

        :param degradation: class, which provides degradation model
        :param network: neural network for gradient descent steps calculation
        :param backprop_through_gradlikelihood: if True, likelihood gradients will are included to autograd graph
        :param num_steps: number of optimizer steps to perform for restoration
        """
        super().__init__()
        self.degradation = degradation
        self.backbone = optimizer_network
        self.num_steps = num_steps
        self.backprop_through_gradlikelihood = backprop_through_gradlikelihood
        if projection_function is None:
            self.projection_function = IdentityProjection()
        else:
            self.projection_function = projection_function

    def perform_step(self, latent_images: th.Tensor, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method performs a minimization step on objective.

        :param latent_images: batch of latent images of shape [B, C1, H1, W1], which should be updated
        :param degraded_images: batch of degraded images of shape [B, C2, H2, W2]
        :return: updated images of shape [B, C, H, W]
        """
        input_gradients = self.degradation.grad_likelihood(degraded_images, latent_images,
                                                           self.backprop_through_gradlikelihood)
        assert input_gradients.shape == latent_images.shape
        network_input = th.cat((latent_images, input_gradients), dim=1)
        update_step = self.backbone(network_input)
        return latent_images + update_step

    def forward(self, degraded_images: th.Tensor) -> List[th.Tensor]:
        """
        This method performs a forward pass on input degraded images.

        :param degraded_images: batch of degraded images of shape [B, C, H, W] needed for restoration
        :return: list of restored images after each step
        """
        return self.restore(degraded_images, track_updates_history=True)
