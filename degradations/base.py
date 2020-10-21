import abc
from typing import Callable, Union

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DegradationBase:
    """
    This is a base class for all degradations
    """
    def __init__(self, likelihood_loss: Callable) -> None:
        """
        Initializing everything that is needed to perform degradation

        :param likelihood_loss: callable function to compute loss, should return a tensor of size 1
        """
        self.likelihood_loss = likelihood_loss

    @abc.abstractmethod
    def degrade(self, images: th.Tensor) -> th.Tensor:
        """
        This method performs degradation of input data, assuming some degradation model.
        Only deterministic part of degradation routine is implemented by this method.
        IMPORTANT: This method should use only differentiable operations.

        :param images: input batch of images [B, C1, H1, W1], which should be degraded
        :return: batch of degraded images [B, C2, H2, W2]
        """
        pass

    def likelihood(self, degraded_images: th.Tensor, latent_images: th.Tensor) -> th.Tensor:
        """
        Method to compute likelihood (data fidelity) term of the total energy
        IMPORTANT: This method should return tensor of size 1.

        :param degraded_images: batch of degraded images
        :param latent_images: batch of current latent images estimate
        :return: data fidelity between degraded image and current estimate
        """
        return self.likelihood_loss(self.degrade(latent_images), degraded_images)

    def grad_likelihood(self, degraded_images: th.Tensor, latent_images: th.Tensor) -> th.Tensor:
        """
        Method, that computes gradient of a likelihood function w.r.t. current latent images estimate.
        General case - computing gradients using autograd mechanics.

        :param degraded_images: batch of degraded images
        :param latent_images: batch of current latent images estimate
        :return: gradient of data fidelity between degraded images and current latent estimates
        """
        latent_images.requires_grad = True
        data_fidelity = self.likelihood(degraded_images, latent_images)
        grad = th.autograd.grad(data_fidelity, latent_images)[0]
        latent_images.requires_grad = False
        return grad

    def init_latent_images(self, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method is used to init latent images from degraded ones for the first restoration step.

        :param degraded_images: batch of images of shape [B, C1, H1, W1] to create latent ones
        :return: initialized latent images of shape [B, C2, H2, W2]
        """
        return degraded_images

    def degrade_ground_truth(self, gt_images: th.Tensor) -> th.Tensor:
        """
        Auxiliary method to perform synthetic degradation of ground truth images, assuming known degradation model
        IMPORTANT: implementation for the case, when degradation is deterministic. May contain non-differentiable ops.

        :param gt_images: batch of images [B, C1, H1, W1] to perform a synthetic degradation
        :return: batch of degraded images [B, C2, H2, W2]
        """
        pass

    def simulate(self, images: th.Tensor, **kwargs):
        """
        This method simulates the whole degradation pipeline, including its stochastic (noise) part.
        Current implementation assumes that degradation is deterministic, override if it is not the case.

        :param images: batch of input images of shape [B, C1, H1, W1] to be degraded
        :param kwargs: auxiliary parameters, required to properly simulate degradation
        :return: degraded batch of images of shape [B, C2, H2, W2]
        """
        return self.degrade(images)


class LinearDegradationBase(DegradationBase):
    """
    This is a base class for all linear degradations of the form y = Ax + n, where n - i.i.d. Gaussian noise
    """
    def __init__(self, likelihood_loss: Callable = F.mse_loss) -> None:
        """
        Initializing everything that is needed to perform a linear degradation

        :param likelihood_loss: callable function to compute loss, should return a tensor of size 1
        """
        super().__init__(likelihood_loss)

    @abc.abstractmethod
    def linear_transform(self, latent_images: th.Tensor) -> th.Tensor:
        """
        Method which performs a linear degradation Ax on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1], which should be degraded
        :return: batch of degraded images [B, C2, H2, W2]
        """
        pass

    def linear_transform_transposed(self, latent_images: th.Tensor) -> th.Tensor:
        """
        Method which performs a transposed linear degradation A^T x on batch of input images x

        :param latent_images: input batch of images [B, C1, H1, W1]
        :return: batch of degraded images [B, C2, H2, W2]
        """
        pass

    def degrade(self, images: th.Tensor) -> th.Tensor:
        """
        This method performs degradation of input data, assuming some degradation model.
        IMPORTANT: This method should use only differentiable operations.

        :param images: input batch of images [B, C1, H1, W1], which should be degraded
        :return: batch of degraded images [B, C2, H2, W2]
        """
        return self.linear_transform(images)

    def degrade_ground_truth(self, gt_images: th.Tensor, std_min: float = 1, std_max: float = 5) -> th.Tensor:
        """
        Auxiliary method to perform synthetic degradation of ground truth images, assuming known linear
        degradation model and zero mean i.i.d. Gaussian noise

        :param gt_images: batch of images [B, C1, H1, W1] to perform a synthetic degradation
        :param std_min: minimal value of standard deviation of noise
        :param std_max: maximal value of standard deviation of noise
        :return: batch of degraded images [B, C2, H2, W2]
        """
        degraded = self.linear_transform(gt_images)
        std = th.rand((gt_images.shape[0], 1, 1, 1), dtype=gt_images.dtype, device=gt_images.device)
        std = std*(std_max - std_min) + std_min
        degraded += th.randn_like(degraded)*std
        return degraded

    def init_latent_images(self, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method is used to init latent images from degraded ones for the first restoration step.
        For linear degradation the transposed operation is usually used as approximation of inverse one.

        :param degraded_images: batch of images of shape [B, C1, H1, W1] to create latent ones
        :return: initialized latent images of shape [B, C2, H2, W2]
        """
        return self.linear_transform_transposed(degraded_images)

    def simulate(self, images: th.Tensor, noise_std: Union[float, th.Tensor]):
        """
        This method simulates the whole degradation pipeline, including additive noise.

        :param images: batch of input images of shape [B, C1, H1, W1] to be degraded
        :param noise_std: standard deviations of noise, float or tensor of shape [], [1] or [B]
        :return: degraded batch of images of shape [B, C2, H2, W2]
        """
        degraded = self.degrade(images)
        noise = th.randn_like(degraded)
        if isinstance(noise_std, float):
            noise *= noise_std
        elif isinstance(noise_std, th.Tensor):
            noise_std = noise_std.to(degraded)
            if noise_std.ndim == 0 or (noise_std.ndim == 1 and noise_std.shape[0] == 1):
                noise *= noise_std
            elif noise_std.ndim == 1 and noise_std.shape[0] == degraded.shape[0]:
                noise *= noise_std[:, None, None, None]
            else:
                raise ValueError(f'Expected standard deviations to have shapes [], [1] or [B], but received tensor '
                                 f'with shape {noise_std.shape}')
        else:
            raise ValueError(f'Expected input parameter noise_std to be either float or torch.Tensor, '
                             f'but given {noise.__class__}')
        return degraded + noise


class NetworkDegradationBase(DegradationBase):
    """
    This class represents degradation, approximated by some neural network (NN)
    """
    def __init__(self, degradation_network: nn.Module, likelihood_loss: Callable,
                 device: Union[str, th.device] = 'cpu') -> None:
        super().__init__(likelihood_loss)
        self.degradation_network = degradation_network.to(device=device)
        self.degradation_network.train(False)
        for param in self.degradation_network.parameters():
            param.requires_grad = False

    def degrade(self, images: th.Tensor) -> th.Tensor:
        """
        This method performs degradation of input data, assuming that degradation model is approximated by NN.

        :param images: input batch of images [B, C1, H1, W1], which should be degraded
        :return: batch of degraded images [B, C2, H2, W2]
        """
        return self.degradation_network(self._normalize_images(images))

    @staticmethod
    def _normalize_images(images: th.Tensor) -> th.Tensor:
        """
        Method, which normalizes images before passing it to neural network.

        :param images: batch of images of shape [B, C, H, W] to normalize
        :return: batch of normalized images of shape [B, C, H, W]
        """
        return images

    @staticmethod
    def _denormalize_images(images: th.Tensor) -> th.Tensor:
        """
        Method, which denormalizes images after neural network for visualization purposes.

        :param images: batch of images of shape [B, C, H, W] to denormalize
        :return: batch of denormalized images of shape [B, C, H, W]
        """
        return images
