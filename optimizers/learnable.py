from .base import OptimizerBase
import torch.nn as nn
import torch as th
from typing import List, Callable, Union
from degradations import DegradationBase
from functions.projections import IdentityProjection


class LearnedGradientDescentOptimizer(nn.Module, OptimizerBase):
    """
    Class which implements learnable gradient descent optimizer as a recurrent convolutional neural network.
    """
    backbone: nn.Module
    backprop_through_gradlikelihood: bool

    def __init__(self, degradation: DegradationBase, optimizer_network: nn.Module, num_steps: int,
                 projection_function: Callable = None, backprop_through_gradlikelihood: bool = False,
                 hidden_state_network: nn.Module = None, device: Union[th.device, str] = 'cpu') -> None:
        """
        Initialize a learnable gradient descent optimizer

        :param degradation: class, which provides degradation model
        :param optimizer_network: neural network for gradient descent steps calculation
        :param num_steps: number of optimizer steps to perform for restoration
        :param backprop_through_gradlikelihood: if True, likelihood gradients will are included to autograd graph
        :param hidden_state_network: neural network used to calculate running hidden state of lgd optimizer
        :param device: device to run optimization and training
        """
        super().__init__()
        self.degradation = degradation
        self.backbone = optimizer_network.to(device=device)
        self.num_steps = num_steps
        self.backprop_through_gradlikelihood = backprop_through_gradlikelihood
        if projection_function is None:
            self.projection_function = IdentityProjection()
        else:
            self.projection_function = projection_function
        self.hidden_state_backbone = hidden_state_network
        if hidden_state_network is not None:
            self.hidden_state_backbone = self.hidden_state_backbone.to(device=device)

    def perform_step(self, latent_images: th.Tensor, degraded_images: th.Tensor, hidden_state: th.Tensor = None
                     ) -> th.Tensor:
        """
        This method performs a minimization step on objective.

        :param latent_images: batch of latent images of shape [B, C1, H1, W1], which should be updated
        :param degraded_images: batch of degraded images of shape [B, C2, H2, W2]
        :param hidden_state: batch of hidden states of shape [B, Ch. H1, W1] to use in calculating update step
        :return: updated images of shape [B, C, H, W]
        """
        input_gradients = self.degradation.grad_likelihood(degraded_images, latent_images,
                                                           self.backprop_through_gradlikelihood)
        assert input_gradients.shape == latent_images.shape
        network_input = th.cat((latent_images, input_gradients), dim=1)

        if self.hidden_state_backbone is not None:
            assert hidden_state is not None
            hidden_state = self.backbone(th.cat((network_input, hidden_state), dim=1))
            network_input = th.cat((network_input, hidden_state))

        update_step = self.backbone(network_input)
        return latent_images + update_step, hidden_state

    def restore(self, degraded_images: th.Tensor, track_updates_history: bool = False
                ) -> Union[List[th.Tensor], th.Tensor]:
        """
        This method performs restoration of input degraded images.

        :param degraded_images: batch of degraded images of shape [B, C, H, W] needed for restoration
        :param track_updates_history: if set to True, latent images after each step are not discarded
        and a list of images from all updates is returned instead of last latent estimate
        :return: restored images of shape [B, C, H, W]
        """
        if track_updates_history:
            history = []
        latent_images = self.degradation.init_latent_images(degraded_images)
        if self.hidden_state_backbone is not None:
            hidden_state = th.zeros_like(latent_images)
        else:
            hidden_state = None
        for i in range(self.num_steps):
            latent_images, hidden_state = self.perform_step(latent_images, degraded_images, hidden_state)
            latent_images = self.project(latent_images)
            if track_updates_history:
                history.append(latent_images)
        if track_updates_history:
            return history
        else:
            return latent_images

    def forward(self, degraded_images: th.Tensor) -> List[th.Tensor]:
        """
        This method performs a forward pass on input degraded images.

        :param degraded_images: batch of degraded images of shape [B, C, H, W] needed for restoration
        :return: list of restored images after each step
        """
        return self.restore(degraded_images, track_updates_history=True)