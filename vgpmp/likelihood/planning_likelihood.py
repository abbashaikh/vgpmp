# likelihood.py (sketch)
import tensorflow as tf
import gpflow

DTYPE = tf.float64

class PlanningLikelihood(gpflow.likelihoods.Likelihood):
    """
    Expects fmean,fvar at support times; computes E_q[log p(y|f)]
    where log p encodes negative penalty terms. We'll estimate with MC.
    """
    def __init__(self, latent_dim: int, obs_scale: float = 1.0, temperature: float = 1.0):
        super().__init__(
            input_dim=None,
            latent_dim=latent_dim,
            observation_dim=None,
        )
        self.obs_scale = gpflow.Parameter(obs_scale, transform=gpflow.utilities.positive())
        self.temperature = gpflow.Parameter(temperature, transform=gpflow.utilities.positive())

        self.center = tf.constant((5.0, 5.5), dtype=DTYPE)
        self.radius = tf.constant(1.0, dtype=DTYPE)
                
        self.grid_size = tf.constant(10.0, dtype=DTYPE)
        self.epsilon = tf.constant(0.1, dtype=DTYPE)
        self.sigma_obs = tf.constant(0.5, dtype=DTYPE)
        self.sigma_box = tf.constant(1.0, dtype=DTYPE)
        self.hinge_softness = tf.constant(0.5, dtype=DTYPE)
        self.enforce_box = False

    def _hinge(self, x: tf.Tensor) -> tf.Tensor:
        s = tf.cast(self.hinge_softness, x.dtype)
        return tf.nn.softplus(x / s) * s
    
    def _circle_signed_distance(self, Pxy: tf.Tensor) -> tf.Tensor:
        """
        Pxy: [S, K, 2]; returns sdist: [S, K], positive outside circle.
        """
        d = tf.norm(Pxy - self.center, axis=-1)  # [S, K]
        return d - self.radius
    
    def _box_violation(self, Pxy: tf.Tensor) -> tf.Tensor:
        """
        Pxy: [S, N, 2]; returns nonnegative violation per time: [S, K]
        """
        x = Pxy[..., 0]  # shape (S, K)
        y = Pxy[..., 1]  # shape (S, K)
        L = self.grid_size

        lower = self._hinge(-x) + self._hinge(-y)              # below 0
        upper = self._hinge(x - L) + self._hinge(y - L)        # above L
        return lower + upper
    
    def _penalty(self, fsamp):
        Pxy = tf.stack([fsamp[..., 0], fsamp[..., 1]], axis=-1)
        Pxy = tf.cast(Pxy, dtype=fsamp.dtype)

        sdist = self._circle_signed_distance(Pxy)                               # shape (S,K)
        penetration = self._hinge(tf.cast(self.epsilon, dtype=fsamp.dtype) - sdist)
        coll = (penetration / self.sigma_obs) ** 2

        if self.enforce_box:
            vbox = self._box_violation(Pxy)                                     # shape (S, K)
            box = (vbox / self.sigma_box)**2
        else:
            box  = tf.zeros_like(coll)

        return coll + box   # [S, K], nonnegative
    
    def _variational_expectations(self, fmean, fvar, Y=None):
        """
        Monte Carlo: draw S samples from q(f) at the batch’s support times
        Input:
            fmean: shape (K, dof)
            fvar: shape (dof, K, K)
        Output:
            expected_log_likelihood: shape (K,)
        """
        S = 8

        samples_shape = tf.concat([
            tf.constant([S], dtype=tf.int32),
            tf.shape(fmean, out_type=tf.int32)
        ], axis=0)
        eps = tf.random.normal(
            shape=samples_shape,
            dtype=fmean.dtype
        )                                                                       # shape (S, K, dof)

        # create PD cholesky
        j = tf.cast(gpflow.config.default_jitter(), fvar.dtype)
        L = tf.linalg.cholesky(
            0.5* (fvar + tf.linalg.matrix_transpose(fvar)) +
            j * tf.eye(tf.shape(fvar)[-1], dtype=fvar.dtype)[None, :, :]
        )                                                                       # shape (dof, K, K)

        fsamp = tf.transpose(fmean, [1, 0])[None, :, :] + (
            tf.transpose(
                tf.linalg.matmul(
                    tf.transpose(eps, [2, 0, 1]),
                    tf.transpose(L, [0, 2, 1])
                ), [1, 0, 2]
            )
        )                                                                       # shape (S, dof, K)
        fsamp = tf.transpose(fsamp, [0, 2, 1])                                  # shape (S. K, dof)

        penalty = self._penalty(fsamp) * self.obs_scale
        penalty_clipped = tf.clip_by_value(penalty, 0.0, tf.cast(1e3, penalty.dtype))
        loglik = - penalty_clipped / (self.temperature + 1e-9)
        return tf.reduce_mean(loglik, axis=0)                                   # shape (K,)

    def _log_prob(self, F, Y):
        raise NotImplementedError("PlanningLikelihood does not define pointwise log_prob.")

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError("PlanningLikelihood does not provide predictive log density.")

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError("PlanningLikelihood does not provide predictive moments.")
