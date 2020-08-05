import numpy as np
import torch
from gpytorch import settings
from gpytorch.distributions.distribution import Distribution
from gpytorch.lazy import DiagLazyTensor, LazyTensor, delazify, lazify
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from pyro.distributions.multivariate_studentt import (
    MultivariateStudentT as PyroMultivariateStudentT,
)
from scipy import stats
from torch.distributions.utils import lazy_property


class MultivariateStudentT(PyroMultivariateStudentT, Distribution):
    def __init__(self, mean, covariance_matrix, nu, data_num, validate_args=False):
        self._islazy = isinstance(mean, LazyTensor) or isinstance(covariance_matrix, LazyTensor)
        if self._islazy:
            if validate_args:
                ms = mean.size(-1)
                cs1 = covariance_matrix.size(-1)
                cs2 = covariance_matrix.size(-2)
                if not (ms == cs1 and ms == cs2):
                    raise ValueError(
                        f"Wrong shapes in {self._repr_sizes(mean, covariance_matrix)}"
                    )
            self.nu = nu

            self.data_num = data_num
            self.df = nu + data_num
            self.loc = mean
            self._covar = covariance_matrix
            self.__unbroadcasted_scale_tril = None
            self._validate_args = validate_args
            batch_shape = _mul_broadcast_shape(self.loc.shape[:-1], covariance_matrix.shape[:-2])
            event_shape = self.loc.shape[-1:]
            # TODO: Integrate argument validation for LazyTensors into torch.distribution validation logic
            super(Distribution, self).__init__(batch_shape, event_shape, validate_args=False)
        else:
            super().__init__(
                loc=mean, covariance_matrix=covariance_matrix, validate_args=validate_args
            )

    @property
    def _unbroadcasted_scale_tril(self):
        if self.islazy and self.__unbroadcasted_scale_tril is None:
            # cache root decoposition
            ust = delazify(self.lazy_covariance_matrix.cholesky())
            self.__unbroadcasted_scale_tril = ust
        return self.__unbroadcasted_scale_tril

    @_unbroadcasted_scale_tril.setter
    def _unbroadcasted_scale_tril(self, ust):
        if self.islazy:
            raise NotImplementedError(
                "Cannot set _unbroadcasted_scale_tril for lazy MVN distributions"
            )
        else:
            self.__unbroadcasted_scale_tril = ust

    def expand(self, batch_size):
        new_loc = self.loc.expand(torch.Size(batch_size) + self.loc.shape[-1:])
        new_covar = self._covar.expand(torch.Size(batch_size) + self._covar.shape[-2:])
        res = self.__class__(new_loc, new_covar, self.nu, self.data_num)
        return res

    def confidence_region(self):
        """
        Returns 2 standard deviations above and below the mean.

        :rtype: (torch.Tensor, torch.Tensor)
        :return: pair of tensors of size (b x d) or (d), where
            b is the batch size and d is the dimensionality of the random
            variable. The first (second) Tensor is the lower (upper) end of
            the confidence region.
        """
        quantiles = (2.5, 97.5)
        mean = self.mean
        quantiles = [
            stats.t.ppf(q / 100.0, self.nu + self.data_num) * self.stddev + mean for q in quantiles
        ]
        return quantiles

    @staticmethod
    def _repr_sizes(mean, covariance_matrix):
        return f"MultivariateStudentT(loc: {mean.size()}, scale: {covariance_matrix.size()})"

    @lazy_property
    def covariance_matrix(self):
        if self.islazy:
            return self._covar.evaluate()
        else:
            return super().covariance_matrix

    # def get_base_samples(self, sample_shape=torch.Size()):
    #     """Get i.i.d. standard Normal samples (to be used with rsample(base_samples=base_samples))"""
    #     with torch.no_grad():
    #         shape = self._extended_shape(sample_shape)
    #         base_samples = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
    #     return base_samples

    @lazy_property
    def lazy_covariance_matrix(self):
        """
        The covariance_matrix, represented as a LazyTensor
        """
        if self.islazy:
            return self._covar
        else:
            return lazify(super().covariance_matrix)

    def log_prob(self, value):
        # log likelihood をここに書く．
        if settings.fast_computations.log_prob.off():
            return super().log_prob(value)

        if self._validate_args:
            self._validate_sample(value)

        mean, covar = self.loc, self.lazy_covariance_matrix
        diff = value - mean

        # Repeat the covar to match the batch shape of diff
        if diff.shape[:-1] != covar.batch_shape:
            if len(diff.shape[:-1]) < len(covar.batch_shape):
                diff = diff.expand(covar.shape[:-1])
            else:
                padded_batch_shape = (
                    *(1 for _ in range(diff.dim() + 1 - covar.dim())),
                    *covar.batch_shape,
                )
                covar = covar.repeat(
                    *(
                        diff_size // covar_size
                        for diff_size, covar_size in zip(diff.shape[:-1], padded_batch_shape)
                    ),
                    1,
                    1,
                )

        # Get log determininat and first part of quadratic form
        # ガウス過程青本 P.90
        inv_quad, logdet = covar.inv_quad_logdet(inv_quad_rhs=diff.unsqueeze(-1), logdet=True)

        log_marginal = 0.5 * (
            -self.data_num * torch.log(self.nu * np.pi)
            - logdet
            - (self.nu + self.data_num) * torch.log(1 + inv_quad / self.nu)
        )
        log_marginal += torch.lgamma((self.nu + self.data_num) / 2) - torch.lgamma(self.nu / 2)

        return log_marginal

    @property
    def variance(self):
        if self.islazy:
            # overwrite this since torch MVN uses unbroadcasted_scale_tril for this
            diag = self.lazy_covariance_matrix.diag()
            diag = diag.view(diag.shape[:-1] + self._event_shape)
            return diag.expand(self._batch_shape + self._event_shape)
        else:
            return super().variance

    def __add__(self, other):
        if isinstance(other, MultivariateStudentT):
            return self.__class__(
                mean=self.mean + other.mean,
                covariance_matrix=(self.lazy_covariance_matrix + other.lazy_covariance_matrix),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.mean + other, self.lazy_covariance_matrix)
        else:
            raise RuntimeError(
                "Unsupported type {} for addition w/ MultivariateStudentT".format(type(other))
            )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        if not (isinstance(other, int) or isinstance(other, float)):
            raise RuntimeError("Can only multiply by scalars")
        if other == 1:
            return self
        return self.__class__(
            mean=self.mean * other, covariance_matrix=self.lazy_covariance_matrix * (other ** 2)
        )

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        rest_idx = idx[:-1]
        last_idx = idx[-1]
        new_mean = self.mean[idx]

        if len(idx) <= self.mean.dim() - 1 and (Ellipsis not in rest_idx):
            new_cov = self.lazy_covariance_matrix[idx]
        elif len(idx) > self.mean.dim():
            raise IndexError(f"Index {idx} has too many dimensions")
        else:
            # In this case we know last_idx corresponds to the last dimension
            # of mean and the last two dimensions of lazy_covariance_matrix
            if isinstance(last_idx, int):
                new_cov = DiagLazyTensor(self.lazy_covariance_matrix.diag()[(*rest_idx, last_idx)])
            elif isinstance(last_idx, slice):
                new_cov = self.lazy_covariance_matrix[(*rest_idx, last_idx, last_idx)]
            elif last_idx is (...):
                new_cov = self.lazy_covariance_matrix[rest_idx]
            else:
                new_cov = self.lazy_covariance_matrix[
                    (*rest_idx, last_idx, slice(None, None, None))
                ][..., last_idx]
        return self.__class__(mean=new_mean, covariance_matrix=new_cov)
