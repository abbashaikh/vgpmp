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
from ..utils.linear_algebra import Cholesky, AnchoredCholesky
from ..posterior import GaussMarkovPosterior

class MarkovVGP(GPModel, InternalDataTrainingLossMixin):
    """
    
    """
    def __init__(
        self,
        data: RegressionData,
        mean: tf.Tensor,
        kernel: GaussMarkovKernel,
        likelihood: PlanningLikelihood,
        posterior: GaussMarkovPosterior,
        num_latent_gps: Optional[int] = None,
    ):
        """
        data = (X, Y) contains the input points [N, 1] and the observations [N, P]
        kernel, likelihood, mean_function are appropriate GPflow objects
        """
        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel=kernel, likelihood=likelihood, num_latent_gps=num_latent_gps)

        self.prior_mean = mean

        self.data = data_input_to_tensor(data)
        X_data, _Y_data = self.data
        self.N, self.P = tf.shape(_Y_data)

        self.start_state = _Y_data[0, :]    # shape (P,)
        self.goal_state = _Y_data[-1, :]    # shape (P,)

        self.Kinv_cholesky = self.kernel.Kinv_cholesky
        self.K_cholesky = self.kernel.K_cholesky
        self.anchor_vars = self.kernel.anchor_vars
        
        static_num_data = X_data.shape[0]
        self.num_data = Parameter(static_num_data, shape=[], dtype=tf.int32, trainable=False)

        self.q_mean_free = Parameter(
            self.prior_mean[1:-1, :],
            shape=(static_num_data - 2, self.num_latent_gps),
        )

        #TODO: introduce whitening of prior
        initial_S_chol = self.K_cholesky
        anchored_S_cholesky = AnchoredCholesky.build_from_unconstrained(
            initial_S_chol,
            anchor_vars=self.anchor_vars
        )
        self.anchored_q_cov_elems = Parameter(
            anchored_S_cholesky.flat_tensor,
            transform=None,
        )

        self.posterior = posterior

    @property
    def q_mean(self):
        conditioned_mu = tf.concat([
            self.start_state[tf.newaxis, :],
            self.q_mean_free,
            self.goal_state[tf.newaxis, :]
        ], axis=0)
        return conditioned_mu   # shape (N, P)
    
    @property
    def q_cov(self):
        anchored_q_cholesky = AnchoredCholesky.build_from_flat_tensor(
            N=self.N, P=self.P,
            flat_tensor=self.anchored_q_cov_elems,
            anchor_vars=self.anchor_vars
        )
        return Cholesky.build_from_anchored_cholesky(anchored_q_cholesky)
    
    def elbo(self, nsamples: int = 8):
        r"""
        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(\mathbf f) = N(\mathbf f \,|\, \boldsymbol \mu, \boldsymbol \Sigma)
        """
        #TODO: verify KL uses cholesky of Kinv and not Kinv
        KL = gauss_markov_kl(self.q_mean, self.q_cov, self.Kinv_cholesky)

        var_exp = self.likelihood.variational_expectations(
            self.q_mean,
            self.q_cov,
            method="gauss_hermite",
            order=10,
        )   # shape (N,)

        return tf.reduce_sum(var_exp) - KL
    
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()
    
    def predict_f(self, Xnew: InputData) -> MeanAndVariance:
        mean, variance = self.posterior.predict_f(
            X_query=Xnew,
            fmean=self.q_mean,
            fvar=self.q_cov,
            K_cholesky=self.K_cholesky
        )
        return mean, variance