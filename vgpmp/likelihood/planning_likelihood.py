from typing import List

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_float, default_jitter

from ..posterior import GaussMarkovPosterior
from ..utils.linear_algebra import Cholesky

SQRT_PI = 1.77245385091
SQRT_TWO = 1.41421356237

class PlanningLikelihood():
    """
    Expects fmean,fvar at support times; computes E_q[log p(y|f)]
    where log p encodes negative penalty terms. We'll estimate with MC.
    """
    def __init__(
        self,
        dof:int = 2,
        obstacle_center=(5.0, 5.0),
        obstacle_radius=2.0,
        grid_size=10.0,
        epsilon: float = 0.1,
        sigma_obs: float = 0.1,
        hinge_softness: float = 0.5,
    ):
        self.dof = dof

        self.center = tf.constant(obstacle_center, dtype=default_float())
        self.radius = tf.constant(obstacle_radius, dtype=default_float())
        self.grid_size = tf.constant(grid_size, dtype=default_float())

        self.epsilon = tf.constant(epsilon, dtype=default_float())
        self.sigma_obs = tf.constant(sigma_obs, dtype=default_float())
        self.hinge_softness = tf.constant(hinge_softness, dtype=default_float())
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
    
    def _gauss_hermite_params(self, order: int):
        # nodes and weights for e^{-x^2}
        xi, wi = np.polynomial.hermite.hermgauss(order)
        xi = tf.convert_to_tensor(xi, dtype=default_float())    # shape (order,)
        wi = tf.convert_to_tensor(wi, dtype=default_float())    # shape (order,)
        norm = tf.cast(SQRT_PI, dtype=default_float())
        return xi, wi / norm    
    
    def _sample_posterior(self, fmean: tf.Tensor, fvar: Cholesky, nsamples: int):
        """
        Input:
            fmean: 
            cov: 
        """
        P = tf.shape(fmean)[-1]
        if tf.equal(tf.rank(fmean), 2):
            N = tf.shape(fmean)[0]
        else:
            N = 1

        # calculate choleskies of S_{kk}: C_{kk}
        C_list = []
        for k in range(N):
            cov_cholesky_row_k = fvar.get_row(k)
            B_k = tf.concat(cov_cholesky_row_k, axis=1)         # shape (P, k+1*P)
            S_kk = tf.matmul(B_k, B_k, transpose_b=True)
            # cholesky of S_{kk}
            C_k = tf.linalg.cholesky(S_kk)                      # shape (P,P)
            C_list.append(C_k)

        # sample z_i from a standard normal
        z_blocks = [tf.random.normal([nsamples, P], dtype=default_float()) for _ in range(N)]
        # y_k ~ normal(0, S_kk)
        y_blocks = [tf.matmul(zi, C_kk, transpose_b=True) for zi, C_kk in zip(z_blocks, fvar)]
        y_samples = tf.stack(y_blocks, axis=1)          # shape (nsamples, N, P)
        # y_k ~ normal(m_k, S_kk)
        y_samples = y_samples + fmean[tf.newaxis, ...]   # shape (nsamples, N, P)
        return y_samples
    
    def variational_expectations(self, fmean: tf.Tensor, fvar: Cholesky, method: str = "gauss_hermite", order: int = 10, nsamples: int = 8, Y=None):
        """
        Gauss-Hermite of order K
        Input:
            fmean: shape (N, P)
            fvar: of type Cholesky

        Monte-carlo with nsamples
        Input:
            fmean: shape (N, P)
            fvar: of type Cholesky
        """
        if method=="gauss_hermite":
            N = tf.shape(fmean)[0]

            xi, wi = self._gauss_hermite_params(order)              # shape (K,)
            X1, X2 = tf.meshgrid(xi, xi, indexing='ij')             # shape (K,K)
            W2 = (wi[:, None] * wi[None, :])                        # shape (K,K)

            # z = [X1, X2]
            Z = tf.stack([X1, X2], axis=-1)                         # shape (K,K,2)

            # calculate choleskies of S_{kk}: C_{kk}
            C_list = []
            for k in range(N):
                cov_cholesky_row_k = fvar.get_row(k)
                B_k = tf.concat(cov_cholesky_row_k, axis=1)         # shape (P, k+1*P)
                S_kk = tf.matmul(B_k, B_k, transpose_b=True)
                # cholesky of S_{kk}
                C_k = tf.linalg.cholesky(S_kk)                      # shape (P,P)
                C_list.append(C_k)
            C = tf.stack(C_list, axis=0)                            # shape (N,P,P)


            # f = mu + sqrt(2) L z
            f = tf.einsum('nij,kkj->nkk i', C[:, :self.dof, :self.dof], Z)  # [N,K,K,dof]
            f = fmean[:, None, None, :self.dof] + tf.cast(SQRT_TWO, dtype=default_float()) * f

            # Signed distance to circle: ||f - center|| - radius
            diff   = f - self.center[None, None, None, :]                   # [N,K,K,dof]
            sdist  = tf.norm(diff, axis=-1) - self.radius                   # [N,K,K]

            s = self.hinge_softness
            penetration = tf.nn.softplus((self.epsilon - sdist) / s) * s    # [N,K,K]
            coll = (penetration / self.sigma_obs)**2                        # [N,K,K]

            E_coll = tf.reduce_sum(coll * W2[None, :, :], axis=(1,2))       # [N]
            return E_coll

        elif method=="monte_carlo":
            fsamples = self._sample_posterior(fmean, fvar, nsamples)
            Pxy = fsamples[..., :self.dof]

            # Signed distance to circle: ||f - center|| - radius (per sample)
            diff   = Pxy - self.center[tf.newaxis, tf.newaxis, :]           # [S,N,dof]
            sdist  = tf.norm(diff, axis=-1) - self.radius                   # [S,N]

            penetration = tf.nn.softplus((self.epsilon - sdist) / s) * s    # [S,N]
            coll = (penetration / self.sigma_obs)**2                        # [S,N]

            E_coll = tf.reduce_sum(coll, axis=0)                            # [N]
            return E_coll
        
        else:
            return NotImplementedError("Expected log-likelihood calculation implemented only using \
                                       Gauss-Hermite Quadrature or Monte Carlo")

    def _log_prob(self, F, Y):
        raise NotImplementedError("PlanningLikelihood does not define pointwise log_prob.")

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError("PlanningLikelihood does not provide predictive log density.")

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError("PlanningLikelihood does not provide predictive moments.")
