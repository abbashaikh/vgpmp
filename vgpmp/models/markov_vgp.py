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
from ..posterior import GaussMarkovPosterior
from ..utils.linear_algebra import (
    BlockTriDiagCovariance, BlockBiDiagFactor,
    invert_softplus_diag,
    covariance_from_bidiag_factor, 
)


class MarkovVGP(GPModel, InternalDataTrainingLossMixin):
    def __init__(
        self,
        data: RegressionData,
        kernel: GaussMarkovKernel,
        likelihood: PlanningLikelihood,
        posterior: GaussMarkovPosterior,
        num_latent_gps: Optional[int] = None,
    ):
        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel=kernel, likelihood=likelihood, num_latent_gps=num_latent_gps)

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

        # --- Variational mean: free mid points only (anchors clamped to Y endpoints) ---
        self.q_mean_free = Parameter(
            _Y_data[1:-1, :],
            shape=(static_num_data - 2, static_num_latent),
        )

        # --- Variational covariance parameterization ---
        # Trainable interior diag raw blocks for L: indices 1..N-2 (so count N-2).
        P_static = static_num_latent
        if P_static is None:
            raise ValueError("num_latent_gps must be static to build MarkovVGP covariance parameters.")

        mid_count = static_num_data - 2
        sub_count = static_num_data - 1

        # # Raw params for trainable mids
        eye_block = tf.eye(P_static, dtype=default_float())
        eye_raw = invert_softplus_diag(eye_block)
        init_mid_raw = tf.tile(eye_raw[None, :, :], multiples=[mid_count, 1, 1])
        init_sub = tf.zeros((sub_count, P_static, P_static), dtype=default_float())

        self.L_mid_diags_raw = Parameter(init_mid_raw, shape=(mid_count, P_static, P_static), transform=None)
        self.L_sub_raw = Parameter(init_sub, shape=(sub_count, P_static, P_static), transform=None)

        # Fixed endpoint diag factors (raw, before softplus in _ensure_lower_with_positive_diag)
        anchor_chol = tf.linalg.diag(tf.sqrt(self.anchor_vars))  # (P,P)
        self.L00_fixed_raw = invert_softplus_diag(anchor_chol)  # (P,P)
        self.LNN_fixed_raw = invert_softplus_diag(anchor_chol)  # (P,P)

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
            [self.L00_fixed_raw[None, :, :], self.L_mid_diags_raw, self.LNN_fixed_raw[None, :, :]],
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
    
    def elbo_terms(self, nsamples: int = 8):
        """
        Returns:
        sum_var_exp: scalar
        KL: scalar
        elbo: scalar (= sum_var_exp - KL)
        """
        KL = gauss_markov_kl(self.q_mean, self.q_cov, self.Kinv)

        var_exp = self.likelihood.variational_expectations(
            self.q_mean,
            self.q_cov,
            method="gauss_hermite",
            order=10,
            nsamples=nsamples,
        )  # (N,)

        sum_var_exp = tf.reduce_sum(var_exp)
        elbo = sum_var_exp - KL
        return sum_var_exp, KL, elbo

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
