from degradations import DegradationBase
import torch as th
import torch.nn as nn
import abc
from typing import Tuple, Callable


class OptimizerBase:
    """
    Base class for all optimizers.
    """
    degradation: DegradationBase
    num_steps: int
    prior_function: Callable
    projection_function: Callable
    reg_weight: float

    def __init__(self, degradation: DegradationBase, num_steps: int, prior_function: Callable, prior_weight: float,
                 projection_function: Callable) -> None:
        """
        Initializing all instances, required for optimization.

        :param degradation: function, which provides degradation model
        :param num_steps: number of optimizer steps to perform for restoration
        :param prior_function: function, which is called to calculate images priors
        :param prior_weight: regularization weight to scale the prior value properly (usually is as hyperparameter)
        :param projection_function: function, which is called to explicitly project images to some specific domain
        """
        self.degradation = degradation
        self.num_steps = num_steps
        self.prior_function = prior_function
        self.projection_function = projection_function
        self.reg_weight = prior_weight

    @abc.abstractmethod
    def perform_step(self, latent_images: th.Tensor, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method performs a minimization step on objective.

        :param latent_images: batch of latent images of shape [B, C1, H1, W1], which should be updated
        :param degraded_images: batch of degraded images of shape [B, C2, H2, W2]
        :return: updated images of shape [B, C, H, W]
        """
        pass

    def prior(self, images: th.Tensor) -> th.Tensor:
        """
        This method is used to define image prior function to use in objective.

        :param images: batch of images of shape [B, C, H, W] for prior calculation
        :return: prior value of shape [B]
        """
        return self.reg_weight*self.prior_function(images)

    def objective(self, latent_images: th.Tensor, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method is used to calculate the value of objective function at some point of interest.

        :param latent_images: batch of input images of shape [B, C1, H1, W1], point of interest
        :param degraded_images: batch of degraded images of shape [B, C2, H2, W2], corresponding to a given input
        :return: value of objective function of shape [B]
        """
        likelihood = self.degradation.likelihood(degraded_images, latent_images)
        prior = self.prior(latent_images)
        return likelihood + prior

    def restore(self, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method performs restoration of input degraded images.

        :param degraded_images: batch of degraded images of shape [B, C, H, W] needed for restoration
        :return: restored images of shape [B, C, H, W]
        """
        latent_images = self.degradation.init_latent_images(degraded_images)
        for i in range(self.num_steps):
            latent_images = self.perform_step(latent_images, degraded_images)
            latent_images = self.project_images(latent_images)
        return latent_images

    def project(self, images: th.Tensor) -> th.Tensor:
        """
        This function implements projection of images, which is performed after each restoration step.
        This is sometimes needed to force images to stay in specific domain during optimization.
        For example, if we are working with natural images, we may desire their values to lie within [0, 1] limits.

        :param latent_images: batch of images of shape [B, C, H, W] for projection
        :return: projected images of shape [B, C, H, W]
        """
        return self.projection_function(images)


class CNNOptimizerBase(OptimizerBase, nn.Module):
    """
    Base class for all optimizers.
    """
    degradation: DegradationBase
    latent_image: th.Tensor
    optimizer_network: nn.Module
    def __init__(self, network: nn.Module, num_steps: int) -> None:
        """
        Initialize a CNN optim
        :param network:
        :param num_steps:
        """
        self.optimizer_network = network
        self.num_steps = num_steps
