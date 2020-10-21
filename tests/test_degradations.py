from itertools import product
from typing import Tuple

import pytest
import torch as th
from scipy.signal import convolve2d

from degradations.base import DegradationBase, LinearDegradationBase
from degradations.linear import NoiseDegradation, BlurDegradation, DownscaleDegradation


class TestDegradationBase:
    degradation_class: DegradationBase
    __test__ = False
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)]}

    def init_test_class(self, *args, **kwargs) -> DegradationBase:
        """
        SHOULD ASSIGN TO self.degradation_class: A DEGRADATION CLASS WITH DOUBLE DTYPES
        """
        pass

    @property
    def sizes_iterator(self):
        keys, values = zip(*self.sizes_params.items())
        for bundle in product(*values):
            d = dict(zip(keys, bundle))
            yield d

    def init_degraded_latent_data(self, batch_size: int, channels_number: int, spatial_dims: Tuple[int, int], **kwargs):
        latent = th.rand(batch_size, channels_number, spatial_dims[0], spatial_dims[1]).double()
        degraded = self.degradation_class.degrade_ground_truth(latent)
        return degraded, latent

    def test_likelihood_grad(self, kwargs=None):
        """
        Check whether likelihood is differentiable
        """
        def pass_test(kwargs):
            self.init_test_class(**kwargs)
            degraded, latent = self.init_degraded_latent_data(**kwargs)
            assert degraded.dtype == th.float64 and latent.dtype == th.float64
            latent.requires_grad = True
            assert th.autograd.gradcheck(self.degradation_class.likelihood, (degraded, latent))
        if kwargs is None:
            for kwargs in self.sizes_iterator:
                pass_test(kwargs)
        else:
            pass_test(kwargs)


class TestLinearDegradationBase(TestDegradationBase):
    degradation_class: LinearDegradationBase
    __test__ = False

    def test_transform_transposed(self, kwargs=None):
        """
        Checks correctness of implementation of linear transposed operator, using property of dot product:
        (A x, y) = (x, A^T y)
        """
        def pass_test(kwargs):
            self.init_test_class(**kwargs)
            vec_a = th.rand(kwargs['batch_size'], kwargs['channels_number'],
                            kwargs['spatial_dims'][0], kwargs['spatial_dims'][1]).double()
            Avec_a = self.degradation_class.linear_transform(vec_a)
            vec_b = th.rand_like(Avec_a)
            A_Tvec_b = self.degradation_class.linear_transform_transposed(vec_b)

            prod_1 = self._prod(Avec_a, vec_b)
            prod_2 = self._prod(vec_a, A_Tvec_b)
            assert th.all(th.isclose(prod_1, prod_2))
        if kwargs is None:
            for kwargs in self.sizes_iterator:
                pass_test(kwargs)
        else:
            pass_test(kwargs)

    def _prod(self, x: th.Tensor, y: th.Tensor):
        assert x.dim() == y.dim() == 4
        assert x.shape == y.shape
        return (x*y).sum(dim=(1,2,3))


class TestDegradationNoise(TestLinearDegradationBase):
    degradation_class: LinearDegradationBase
    __test__ = True

    def init_test_class(self, *args, **kwargs):
        self.degradation_class = NoiseDegradation()


class TestDegradationBlur(TestLinearDegradationBase):
    degradation_class: BlurDegradation
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'kernel_size': [4, 5]}

    def init_test_class(self, batch_size, channels_number, spatial_dims, kernel_size=5):
        self.degradation_class = BlurDegradation(th.rand(batch_size, 1, kernel_size, kernel_size).double())

    def test_convolution(self, kwargs=None):
        """
        Checks whether batched convolution is computed correctly - compare it with scipy.signal.convolve2d.
        """
        def pass_test(kwargs):
            self.init_test_class(**kwargs)
            image = th.rand(kwargs['batch_size'], kwargs['channels_number'],
                            kwargs['spatial_dims'][0], kwargs['spatial_dims'][1]).double()
            convolved_th = self.degradation_class._valid_convolve(image, self.degradation_class.kernels)
            kernel = self.degradation_class.kernels
            convolved_true = []
            for batch in range(kwargs['batch_size']):
                colored_image = []
                for channel in range(kwargs['channels_number']):
                    conv = convolve2d(image[batch, channel, :, :].numpy(), kernel[batch, 0, :, :].numpy(), mode='valid')
                    colored_image.append(th.from_numpy(conv))
                convolved_true.append(th.stack(colored_image, dim=0))
            convolved_true = th.stack(convolved_true, dim=0)
            assert th.all(th.isclose(convolved_true, convolved_th))
        if kwargs is None:
            for kwargs in self.sizes_iterator:
                pass_test(kwargs)
        else:
            pass_test(kwargs)


class TestDegradationDownscale(TestDegradationBlur):
    degradation_class: DownscaleDegradation
    __test__ = True
    sizes_params = {'batch_size': [1, 2],
                    'channels_number': [1, 3],
                    'spatial_dims': [(16, 16), (17, 17), (16, 17)],
                    'kernel_size': [4, 5],
                    'scale_factor': [1, 2, 3]}

    def init_test_class(self, batch_size, channels_number, spatial_dims, kernel_size, scale_factor):
        self.degradation_class = DownscaleDegradation(
            scale_factor, th.rand(batch_size, 1, kernel_size, kernel_size).double())

    def test_transform_transposed(self):
        for kwargs in self.sizes_iterator:
            if (kwargs['spatial_dims'][0] - kwargs['kernel_size'] + 1)%kwargs['scale_factor'] == 0 and \
                    (kwargs['spatial_dims'][1] - kwargs['kernel_size'] + 1)%kwargs['scale_factor'] == 0:
                super().test_transform_transposed(kwargs)
            else:
                with pytest.raises(AssertionError):
                    super().test_transform_transposed(kwargs)

    def test_likelihood_grad(self):
        for kwargs in self.sizes_iterator:
            if (kwargs['spatial_dims'][0] - kwargs['kernel_size'] + 1)%kwargs['scale_factor'] == 0 and \
                    (kwargs['spatial_dims'][1] - kwargs['kernel_size'] + 1)%kwargs['scale_factor'] == 0:
                super().test_likelihood_grad(kwargs)
            else:
                with pytest.raises(AssertionError):
                    super().test_likelihood_grad(kwargs)
