from typing import Optional

import tensorflow as tf
from gpflow import Parameter
from gpflow.models import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.config import default_float

from ..base import RegressionData, InputData, MeanAndVariance
from ..kernels import GaussMarkovKernel
from ..likelihood import PlanningLikelihood
from ..kulback_liebler import gauss_markov_kl
from ..utils.linear_algebra import BlockTriDiagCovariance, BlockBiDiagFactor, covariance_from_bidiag_factor
from ..posterior import GaussMarkovPosterior


class MarkovVGP(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: RegressionData,
        mean: tf.Tensor,
        kernel: GaussMarkovKernel,
        likelihood: PlanningLikelihood,
        posterior: GaussMarkovPosterior,
        num_latent_gps: Optional[int] = None,
    ):
        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel=kernel, likelihood=likelihood, num_latent_gps=num_latent_gps)

        self.prior_mean = mean

        self.data = data_input_to_tensor(data)
        X_data, _Y_data = self.data

        # Prefer static ints when available (for Parameter shapes)
        static_num_data = X_data.shape[0]
        static_num_latent = self.num_latent_gps

        self.N = tf.shape(_Y_data)[0]
        self.P = tf.shape(_Y_data)[1]

        self.start_state = _Y_data[0, :]   # (P,)
        self.goal_state = _Y_data[-1, :]   # (P,)

        self.Kinv = self.kernel.Kinv
        self.K = self.kernel.K
        self.anchor_vars = self.kernel.anchor_vars

        if static_num_data is None:
            raise ValueError("X_data must have static length to build MarkovVGP Parameters.")
        self.num_data = Parameter(static_num_data, shape=[], dtype=tf.int32, trainable=False)

        # Variational mean: free mid points only (anchors clamped to Y endpoints)
        self.q_mean_free = Parameter(
            self.prior_mean[1:-1, :],
            shape=(static_num_data - 2, static_num_latent),
        )

        # Variational covariance parameterization:

        # Trainable interior diag raw blocks for L: indices 1..N-2 (so count N-2)
        # Initialize near identity (or use something based on prior scale)
        init_mid = tf.eye(self.P, dtype=default_float())[None, :, :] * tf.ones((self.N-2, 1, 1), dtype=default_float())
        self.L_mid_diags_raw = Parameter(init_mid, transform=None)

        # Trainable sub-diagonal blocks L_{i+1,i} for i=0..N-2 (count N-1)
        init_sub = tf.zeros((self.N-1, self.P, self.P), dtype=default_float())
        self.L_sub_raw = Parameter(init_sub, transform=None)

        # Fixed endpoint diag factors
        self.L00_fixed = tf.linalg.diag(tf.sqrt(self.anchor_vars))  # (P,P)
        self.LNN_fixed = tf.linalg.diag(tf.sqrt(self.anchor_vars))  # (P,P)

        self.posterior = posterior

    @property
    def q_mean(self) -> tf.Tensor:
        return tf.concat(
            [
                self.start_state[tf.newaxis, :],
                self.q_mean_free,
                self.goal_state[tf.newaxis, :],
            ],
            axis=0,
        )  # (N,P)

    @property
    def q_cov(self) -> BlockTriDiagCovariance:
        # Assemble full diags_raw: [L00_fixed, L_mid_diags_raw, LNN_fixed]
        diags_raw = tf.concat(
            [self.L00_fixed[None, :, :], self.L_mid_diags_raw, self.LNN_fixed[None, :, :]],
            axis=0,
        )  # (N,P,P)

        sub_raw = self.L_sub_raw

        L = BlockBiDiagFactor(diags_raw=diags_raw, sub_diags_raw=sub_raw)
        return covariance_from_bidiag_factor(L)

    def elbo(self, nsamples: int = 8) -> tf.Tensor:
        r"""
        ELBO = E_q[ log p(Y|F) ] - KL[q(F) || p(F)]
        """
        KL = gauss_markov_kl(self.q_mean, self.q_cov, self.Kinv)

        var_exp = self.likelihood.variational_expectations(
            self.q_mean,
            self.q_cov,
            method="gauss_hermite",
            order=10,
            nsamples=nsamples,
        )  # (N,)

        return tf.reduce_sum(var_exp) - KL

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def predict_f(self, Xnew: InputData) -> MeanAndVariance:
        mean, variance = self.posterior.predict_f(
            X_query=Xnew,
            fmean=self.q_mean,
            fvar=self.q_cov,
            K_prior=self.K,
        )
        return mean, variance
