import torch as th
import abc

class ParametricFunctionBase:
    """
    Base class to implement parametric functions, i.e. priors
    """
    def __call__(self, *args, **kwargs): return self.call(*args, **kwargs)

    @abc.abstractmethod
    def call(self, input_tensor: th.Tensor) -> th.Tensor:
        """
        Computes some value of function

        :param input_tensor: input to function
        :return: value of function, evaluated on input
        """
        pass
