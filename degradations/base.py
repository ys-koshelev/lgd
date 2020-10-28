import abc
from typing import Callable, Union, Dict, Any

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DegradationBase:
    """
    This is a base class for all degradations
    """
    def __init__(self, likelihood_loss: Callable, device: Union[th.device, str] = 'cpu') -> None:
        """
        Initializing everything that is needed to perform degradation

        :param likelihood_loss: callable function to compute loss, should return a tensor of size 1
        :param device: device to place internal parameters
        """
        self.likelihood_loss = likelihood_loss
        self.device = device

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

    def grad_likelihood(self, degraded_images: th.Tensor, latent_images: th.Tensor,
                        propagate_gradients_through_grad: bool = False) -> th.Tensor:
        """
        Method, that computes gradient of a likelihood function w.r.t. current latent images estimate.
        General case - computing gradients using autograd mechanics.

        :param degraded_images: batch of degraded images
        :param latent_images: batch of current latent images estimate
        :param propagate_gradients_through_grad: if True, gradients will be back propagated through this operation
        :return: gradient of data fidelity between degraded images and current latent estimates
        """
        def func(x):
            return self.likelihood(degraded_images, x)
        if propagate_gradients_through_grad and (degraded_images.requires_grad or latent_images.requires_grad):
            create_graph = True
        else:
            # no need to create useless graph, if none of inputs requires gradient
            create_graph = False
        grad = th.autograd.functional.vjp(func, latent_images, create_graph=create_graph)[1]
        return grad

    def init_latent_images(self, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method is used to init latent images from degraded ones for the first restoration step.

        :param degraded_images: batch of images of shape [B, C1, H1, W1] to create latent ones
        :return: initialized latent images of shape [B, C2, H2, W2]
        """
        return degraded_images

    def init_random_parameters_and_degrade(self, gt_images: th.Tensor, *args, **kwargs) -> th.Tensor:
        """
        Auxiliary method to initize parameters randomly and perform synthetic degradation of ground truth images,
        assuming known degradation model

        :param gt_images: batch of images [B, C1, H1, W1] to perform a synthetic degradation
        :param args, kwargs: inputs used to random sampling of parameters
        :return: batch of degraded images [B, C2, H2, W2]
        """
        self.init_random_parameters(gt_images.shape[0], *args, **kwargs)
        return self.simulate_degradation(gt_images)

    def simulate_degradation(self, images: th.Tensor, **kwargs):
        """
        This method simulates the whole degradation pipeline, including its stochastic (noise) part.
        Current implementation assumes that degradation is deterministic, override if it is not the case.

        :param images: batch of input images of shape [B, C1, H1, W1] to be degraded
        :param kwargs: auxiliary parameters, required to properly simulate degradation
        :return: degraded batch of images of shape [B, C2, H2, W2]
        """
        return self.degrade(images)

    def init_parameters(self, *args, **kwargs) -> None:
        """
        This method allows to change degradation parameters inplace without re-initializing degradation class.

        :param args, kwargs: inputs with new parameters values
        :return: Nothing
        """
        pass

    def init_random_parameters(self, batch_size: int, *args, **kwargs) -> None:
        """
        This method allows to reinitialize degradation parameters randomly without re-initializing degradation class.

        :param batch_size: size of batch to restore, needed to sample different parameters for each image in batch
        :param args, kwargs: arguments, needed to initialize random instances of degradation
        :return: Nothing
        """
        pass


class LinearDegradationBase(DegradationBase):
    """
    This is a base class for all linear degradations of the form y = Ax + n, where n - i.i.d. Gaussian noise
    """
    def __init__(self, noise_std: Union[th.Tensor, float], device: Union[th.device, str] = 'cpu') -> None:
        """
        Initializing everything that is needed to perform a linear degradation

        :param noise_std: standard deviation of i.i.d. additive Gaussian noise
        :param device: device to place internal parameters
        """
        super().__init__(F.mse_loss, device)
        self.noise_std = noise_std

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

    def likelihood(self, degraded_images: th.Tensor, latent_images: th.Tensor) -> th.Tensor:
        """
        Method to compute likelihood (data fidelity) term of the total energy
        IMPORTANT: This method should return tensor of size 1.

        :param degraded_images: batch of degraded images
        :param latent_images: batch of current latent images estimate
        :return: data fidelity between degraded image and current estimate
        """
        return self.likelihood_loss(self.degrade(latent_images)/self._noise_std, degraded_images/self._noise_std)

    @property
    def _noise_std(self):
        if isinstance(self.noise_std, float):
            ret = th.FloatTensor([self.noise_std]).to(device=self.device)
        elif isinstance(self.noise_std, th.Tensor):
            assert self.noise_std.ndim == 0 or self.noise_std.ndim == 1
            ret = self.noise_std.to(device=self.device)
        dim = ret.ndim
        for i in range(4 - dim):
            ret = ret.unsqueeze(-1)
        return ret

    def init_latent_images(self, degraded_images: th.Tensor) -> th.Tensor:
        """
        This method is used to init latent images from degraded ones for the first restoration step.
        For linear degradation the transposed operation is usually used as approximation of inverse one.

        :param degraded_images: batch of images of shape [B, C1, H1, W1] to create latent ones
        :return: initialized latent images of shape [B, C2, H2, W2]
        """
        return self.linear_transform_transposed(degraded_images)

    @staticmethod
    def _add_noise(images: th.Tensor, noise_std: Union[th.Tensor, float]):
        noise = th.randn_like(images)
        if isinstance(noise_std, float):
            noise *= noise_std
        elif isinstance(noise_std, th.Tensor):
            noise_std = noise_std.to(images)
            if noise_std.ndim == 0 or (noise_std.ndim == 1 and noise_std.shape[0] == 1):
                noise *= noise_std
            elif noise_std.ndim == 1 and noise_std.shape[0] == images.shape[0]:
                noise *= noise_std[:, None, None, None]
            else:
                raise ValueError(f'Expected standard deviations to have shapes [], [1] or [B], but received tensor '
                                 f'with shape {noise_std.shape}')
        else:
            raise ValueError(f'Expected input parameter noise_std to be either float or torch.Tensor, '
                             f'but given {noise_std.__class__}')
        return images + noise

    def simulate_degradation(self, images: th.Tensor):
        """
        This method simulates the whole degradation pipeline, including additive noise.

        :param images: batch of input images of shape [B, C1, H1, W1] to be degraded
        :return: degraded batch of images of shape [B, C2, H2, W2]
        """
        assert images.min() >= 0 and images.max() <= 1, "Images have to be in natural representation within range " \
                                                        "[0, 1]"
        degraded = self.degrade(images)
        degraded = self._add_noise(degraded, self.noise_std)
        degraded = th.clamp(degraded, 0, 1)
        return degraded

    def init_random_parameters(self, batch_size: int, noise_std_min: Union[th.Tensor, float],
                               noise_std_max: Union[th.Tensor, float]) -> None:
        """
        This method allows to reinitialize degradation parameters randomly without re-initializing degradation class.

        :param batch_size: size of batch to restore, needed to sample different parameters for each image in batch
        :param noise_std_min: lower bound value of noise standard deviation to use in degradation
        :param noise_std_max: upper bound value of noise standard deviation to use in degradation
        """
        self.noise_std = noise_std_min + th.rand(batch_size)*(noise_std_max - noise_std_min)

    @property
    def params_dict(self) -> Dict[str, th.Tensor]:
        """
        Method, which returns all current degradation parameters as a dict

        :return: dict with parameters to be passed as kwargs
        """
        return {'noise_std': self.noise_std}


class NetworkDegradationBase(DegradationBase):
    """
    This class represents degradation, approximated by some neural network (NN)
    """
    def __init__(self, degradation_network: nn.Module, likelihood_loss: Callable,
                 device: Union[str, th.device] = 'cpu') -> None:
        super().__init__(likelihood_loss, device)
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
        return self._denormalize_images(self.degradation_network(self._normalize_images(images)))

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
