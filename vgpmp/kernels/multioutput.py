import abc
from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf
from check_shapes import check_shape as cs
from check_shapes import check_shapes, inherit_check_shapes
from gpflow import Parameter

from .gauss_markov import GaussMarkov

TensorType = Union[tf.Tensor, tf.Variable, Parameter]

class MultioutputKernel(GaussMarkov):
    """
    Multi Output Kernel class.

    This kernel can represent correlation between outputs of different datapoints.

    The `full_output_cov` argument holds whether the kernel should calculate
    the covariance between the outputs. In case there is no correlation but
    `full_output_cov` is set to True the covariance matrix will be filled with zeros
    until the appropriate size is reached.
    """

    @property
    @abc.abstractmethod
    def num_latent_gps(self) -> int:
        """The number of latent GPs in the multioutput kernel"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def latent_kernels(self) -> Tuple[GaussMarkov, ...]:
        """The underlying kernels in the multioutput kernel"""
        raise NotImplementedError

    @abc.abstractmethod
    @check_shapes(
        "X: [batch..., N, D]",
        "X2: [batch2..., N2, D]",
        "return: [batch..., N, P, batch2..., N2, P] if full_output_cov and (X2 is not None)",
        "return: [P, batch..., N, batch2..., N2] if not full_output_cov and (X2 is not None)",
        "return: [batch..., N, P, N, P] if full_output_cov and (X2 is None)",
        "return: [P, batch..., N, N] if not full_output_cov and (X2 is None)",
    )
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        """
        Returns the correlation of f(X) and f(X2), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param X2: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: cov[f(X), f(X2)]
        """
        raise NotImplementedError

    @abc.abstractmethod
    @check_shapes(
        "X: [batch..., N, D]",
        "return: [batch..., N, P, P] if full_output_cov",
        "return: [batch..., N, P] if not full_output_cov",
    )
    def K_diag(self, X: TensorType, full_output_cov: bool = True) -> tf.Tensor:
        """
        Returns the correlation of f(X) and f(X), where f(.) can be multi-dimensional.

        :param X: data matrix
        :param full_output_cov: calculate correlation between outputs.
        :return: var[f(X)]
        """
        raise NotImplementedError

    @check_shapes(
        "X: [batch..., N, D]",
        "X2: [batch2..., N2, D]",
        "return: [batch..., N, P, batch2..., N2, P] if full_cov and full_output_cov and (X2 is not None)",
        "return: [P, batch..., N, batch2..., N2] if full_cov and (not full_output_cov) and (X2 is not None)",
        "return: [batch..., N, P, N, P] if full_cov and full_output_cov and (X2 is None)",
        "return: [P, batch..., N, N] if full_cov and (not full_output_cov) and (X2 is None)",
        "return: [batch..., N, P, P] if (not full_cov) and full_output_cov and (X2 is None)",
        "return: [batch..., N, P] if (not full_cov) and (not full_output_cov) and (X2 is None)",
    )
    def __call__(
        self,
        X: TensorType,
        X2: Optional[TensorType] = None,
        *,
        full_cov: bool = False,
        full_output_cov: bool = True,
        presliced: bool = False,
    ) -> tf.Tensor:
        if not presliced:
            X, X2 = self.slice(X, X2)
        if not full_cov and X2 is not None:
            raise ValueError(
                "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
            )
        if not full_cov:
            return self.K_diag(X, full_output_cov=full_output_cov)
        return self.K(X, X2, full_output_cov=full_output_cov)
    

class SeparateIndependent(MultioutputKernel, Combination):
    """
    - Separate: we use different kernel for each output latent
    - Independent: Latents are uncorrelated a priori.
    """

    def __init__(self, kernels: Sequence[Kernel], name: Optional[str] = None) -> None:
        super().__init__(kernels=kernels, name=name)

    @property
    def num_latent_gps(self) -> int:
        return len(self.kernels)

    @property
    def latent_kernels(self) -> Tuple[Kernel, ...]:
        """The underlying kernels in the multioutput kernel"""
        return tuple(self.kernels)

    @inherit_check_shapes
    def K(
        self, X: TensorType, X2: Optional[TensorType] = None, full_output_cov: bool = True
    ) -> tf.Tensor:
        rank = tf.rank(X) - 1
        if X2 is None:
            if full_output_cov:
                Kxxs = cs(
                    tf.stack([k.K(X, X2) for k in self.kernels], axis=-1), "[batch..., N, N, P]"
                )
                perm = tf.concat(
                    [
                        tf.range(rank),
                        [rank + 1, rank, rank + 2],
                    ],
                    0,
                )
                return cs(tf.transpose(tf.linalg.diag(Kxxs), perm), "[batch..., N, P, N, P]")
            else:
                return cs(
                    tf.stack([k.K(X, X2) for k in self.kernels], axis=0), "[P, batch..., N, N]"
                )
        else:
            rank2 = tf.rank(X2) - 1
            if full_output_cov:
                Kxxs = cs(
                    tf.stack([k.K(X, X2) for k in self.kernels], axis=-1),
                    "[batch..., N, batch2..., N2, P]",
                )
                perm = tf.concat(
                    [
                        tf.range(rank),
                        [rank + rank2],
                        rank + tf.range(rank2),
                        [rank + rank2 + 1],
                    ],
                    0,
                )
                return cs(
                    tf.transpose(tf.linalg.diag(Kxxs), perm), "[batch..., N, P, batch2..., N2, P]"
                )
            else:
                return cs(
                    tf.stack([k.K(X, X2) for k in self.kernels], axis=0),
                    "[P, batch..., N, batch2..., N2]",
                )

    @inherit_check_shapes
    def K_diag(self, X: TensorType, full_output_cov: bool = False) -> tf.Tensor:
        stacked = cs(tf.stack([k.K_diag(X) for k in self.kernels], axis=-1), "[batch..., N, P]")
        if full_output_cov:
            return cs(tf.linalg.diag(stacked), "[batch..., N, P, P]")
        else:
            return stacked
