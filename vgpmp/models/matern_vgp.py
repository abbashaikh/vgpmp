# models.py
import math
import gpflow
import tensorflow as tf

class MaternVGP(gpflow.models.SVGP):
    """
    SVGP over positions with custom planning likelihood for interior points,
    plus near-hard endpoint observations via a high-precision Gaussian.
    """
    def __init__(self, prior, planning_likelihood, q_mu):
        # prior: MaternPositionPrior
        super().__init__(
            kernel=prior.kernel,
            likelihood=planning_likelihood,
            inducing_variable=prior.inducing_variable,
            num_latent_gps=prior.dof,
            q_mu=q_mu,
            whiten=True,
            q_diag=False
        )
        self.obs_likelihood = prior.obs_likelihood  # tiny variance Gaussian for endpoints

    @tf.function
    def elbo(self, data):
        """
        data = dict with:
          - 'interior': (X_in, Y_in) for planning likelihood (Y_in can be dummies)
          - 'endpoints': (X_ep, Y_ep) exact positions at t0,tN to pin endpoints
        """
        (X_in, _) = data['interior']
        (X_ep, Y_ep) = data['endpoints']

        # (1) expected log-likelihood over interior supports under planning likelihood
        fmean_in, fvar_in = self.predict_f(X_in, full_cov=True, full_output_cov=False)
        # fmean: shape (K, dof); fvar: shape (dof, K, K)
        var_exp_in = self.likelihood._variational_expectations(fmean_in, fvar_in)
        elbo_interior = tf.reduce_sum(var_exp_in)

        # (2) near-hard endpoint constraints using tiny-variance Gaussian
        fmean_ep, fvar_ep = self.predict_f(X_ep, full_cov=False, full_output_cov=False)
        # exact VE for Gaussian:
        var_ep = fvar_ep + self.obs_likelihood.variance  # add noise
        # log N(Y_ep; fmean_ep, var_ep), summed over joints & endpoints
        elbo_endpoints = -0.5*(
            tf.reduce_sum(
                tf.math.log(2.0*tf.constant(math.pi, fmean_ep.dtype)*var_ep)
                + (Y_ep - fmean_ep)**2/var_ep
            )
        )

        # (3) KL on q(u)
        kl = gpflow.kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel,
            self.q_mu, self.q_sqrt, whiten=self.whiten
        )

        # total ELBO
        return elbo_interior + elbo_endpoints - kl
    
    def training_loss_closure(self, data, compile: bool = True):
        """
        Returns a zero-argument callable that computes the current training loss
        for the captured `data`. If `compile=True`, wraps it in @tf.function.
        """
        if compile:
            @tf.function
            def closure():
                return -self.elbo(data)
        else:
            def closure():
                return -self.elbo(data)
        return closure
