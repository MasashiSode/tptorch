import warnings
from copy import deepcopy
from typing import Any, Optional

import pyro
import torch
from gpytorch.lazy import ZeroLazyTensor
from gpytorch.likelihoods import Likelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise, HomoskedasticNoise, Noise
from gpytorch.utils.warnings import GPInputWarning
from pyro.distributions import StudentT
from torch import Tensor

from ..distributions import MultivariateStudentT


class _StudentTLikelihoodBase(Likelihood):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models."""

    def __init__(self, noise_covar: Noise, **kwargs: Any) -> None:

        super().__init__()
        param_transform = kwargs.get("param_transform")
        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformaton, specify a different 'noise_constraint' instead.",
                DeprecationWarning,
            )
        self.noise_covar = noise_covar

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    # def expected_log_prob(
    #     self, target: Tensor, input: MultivariateStudentT, *params: Any, **kwargs: Any
    # ) -> Tensor:
    #     mean, variance = input.mean, input.variance
    #     num_event_dim = len(input.event_shape)

    #     noise = self._shaped_noise_covar(mean.shape, *params, **kwargs).diag()
    #     # Potentially reshape the noise to deal with the multitask case
    #     noise = noise.view(*noise.shape[:-1], *input.event_shape)

    #     res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi)
    #     res = res.mul(-0.5)
    #     if num_event_dim > 1:  # Do appropriate summation for multitask Gaussian likelihoods
    #         res = res.sum(list(range(-1, -num_event_dim, -1)))
    #     return res

    def forward(
        self, function_samples: Tensor, nu, data_num, *params: Any, **kwargs: Any
    ) -> StudentT:

        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        return StudentT(nu + data_num, function_samples, noise.sqrt())

    def log_marginal(
        self,
        observations: Tensor,
        function_dist: MultivariateStudentT,
        *params: Any,
        **kwargs: Any,
    ) -> Tensor:
        marginal = self.marginal(function_dist, *params, **kwargs)
        # We're making everything conditionally independent
        indep_dist = StudentT(marginal.df, marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())

        res = indep_dist.log_prob(observations)

        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(function_dist.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res

    def marginal(
        self, function_dist: MultivariateStudentT, *params: Any, **kwargs: Any
    ) -> MultivariateStudentT:
        mean, covar, nu, data_num = (
            function_dist.mean,
            function_dist.lazy_covariance_matrix,
            function_dist.nu,
            function_dist.data_num,
        )
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar, nu, data_num)

    def __call__(self, input, *args, **kwargs):
        # Conditional
        if torch.is_tensor(input):
            return super().__call__(input, *args, **kwargs)
        # Marginal
        elif any(
            [
                isinstance(input, StudentT),
                isinstance(input, MultivariateStudentT),
                (
                    isinstance(input, pyro.distributions.Independent)
                    and isinstance(input.base_dist, pyro.distributions.Normal)
                ),
            ]
        ):
            return self.marginal(input, *args, **kwargs)
        # Error
        else:
            raise RuntimeError(
                "Likelihoods expects a MultivariateStudentT or StudentT input to make marginal predictions, or a "
                "torch.Tensor for conditional predictions. Got a {}".format(
                    input.__class__.__name__
                )
            )


class StudentTLikelihood(_StudentTLikelihoodBase):
    def __init__(
        self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), **kwargs,
    ):
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)


class FixedNoiseStudentTLikelihood(_StudentTLikelihoodBase):
    """
    A Likelihood that assumes fixed heteroscedastic noise. This is useful when you have fixed, known observation
    noise for each training example.

    Args:
        :attr:`noise` (Tensor):
            Known observation noise (variance) for each training example.
        :attr:`learn_additional_noise` (bool, optional):
            Set to true if you additionally want to learn added diagonal noise, similar to GaussianLikelihood.

    Note that this likelihood takes an additional argument when you call it, `noise`, that adds a specified amount
    of noise to the passed MultivariateStudentT. This allows for adding known observational noise to test data.

    Example:
        >>> train_x = torch.randn(55, 2)
        >>> noises = torch.ones(55) * 0.01
        >>> likelihood = FixedNoiseStudentTLikelihood(noise=noises, learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 2)
        >>> test_noises = torch.ones(21) * 0.02
        >>> pred_y = likelihood(tp_model(test_x), noise=test_noises)
    """

    def __init__(
        self,
        noise: Tensor,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: Optional[torch.Size] = torch.Size(),
        **kwargs: Any,
    ) -> None:
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))

        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = HomoskedasticNoise(
                noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
            )
        else:
            self.second_noise_covar = None

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise + self.second_noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Tensor:
        if self.second_noise_covar is None:
            return 0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: Tensor) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for FixedNoiseGaussianLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)

    def get_fantasy_likelihood(self, **kwargs):
        if "noise" not in kwargs:
            raise RuntimeError("FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg")
        old_noise_covar = self.noise_covar
        self.noise_covar = None
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")
        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])
        fantasy_liklihood.noise_covar = FixedGaussianNoise(
            noise=torch.cat([old_noise, new_noise], -1)
        )
        return fantasy_liklihood

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, ZeroLazyTensor):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                GPInputWarning,
            )

        return res
