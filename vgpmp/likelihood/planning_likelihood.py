from typing import List

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_float, default_jitter

from ..posterior import GaussMarkovPosterior
from ..utils.linear_algebra import Covariance

SQRT_PI = 1.77245385091
SQRT_TWO = 1.41421356237


class PlanningLikelihood:
    """
    Expects fmean,fvar at support times; computes E_q[log p(y|f)]
    where log p encodes negative penalty terms. Estimate with GH or MC.

    NOTE: The implementations below use ONLY the marginal covariances S_kk (diagonal blocks)
    and therefore ignore cross-time correlations in fvar.
    """
    def __init__(
        self,
        dof: int,
        desired_nominal,
        obstacle_center=(5.0, 5.0),
        obstacle_radius=2.0,
        grid_size=10.0,
        epsilon: float = 0.1,
        hinge_softness: float = 0.5,
        sigma_obs: float = 0.02,
        sigma_nominal: float = 0.8,
    ):
        self.dof = dof

        self.center = tf.constant(obstacle_center, dtype=default_float())
        self.radius = tf.constant(obstacle_radius, dtype=default_float())
        self.grid_size = tf.constant(grid_size, dtype=default_float())

        self.epsilon = tf.constant(epsilon, dtype=default_float())
        self.sigma_obs = tf.constant(sigma_obs, dtype=default_float())
        self.hinge_softness = tf.constant(hinge_softness, dtype=default_float())
        self.enforce_box = False
        self.sigma_box = tf.constant(0.1, dtype=default_float())

        self.sigma_nominal = tf.constant(sigma_nominal, dtype=default_float())
        self.tracking_weight = tf.constant(1.0, dtype=default_float())
        self.desired_nominal  = None if desired_nominal  is None else tf.constant(desired_nominal,  dtype=default_float())

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
        Pxy: [S, K, 2]; returns nonnegative violation per time: [S, K]
        """
        x = Pxy[..., 0]
        y = Pxy[..., 1]
        L = self.grid_size

        lower = self._hinge(-x) + self._hinge(-y)
        upper = self._hinge(x - L) + self._hinge(y - L)
        return lower + upper

    def _penalty(self, fsamp: tf.Tensor) -> tf.Tensor:
        """
        fsamp: [S, K, P] (or [S,K,dof] if already reduced)
        returns: [S, K] nonnegative
        """
        Pxy = fsamp[..., :self.dof]
        Pxy = tf.cast(Pxy, dtype=fsamp.dtype)

        sdist = self._circle_signed_distance(Pxy)  # [S,K]
        penetration = self._hinge(tf.cast(self.epsilon, dtype=fsamp.dtype) - sdist)
        coll = (penetration / self.sigma_obs) ** 2

        if self.enforce_box:
            vbox = self._box_violation(Pxy)
            box = (vbox / self.sigma_box) ** 2
        else:
            box = tf.zeros_like(coll)

        return coll + box
    
    def _expected_nominal_tracking_penalty(
        self,
        fmean: tf.Tensor,   # (N,P)
        fvar: Covariance,   # fvar.diags: (N,P,P)
    ) -> tf.Tensor:
        """
        Per-time expected quadratic tracking cost to a nominal trajectory (full state).
        Returns: (N,)

        E[||x_k - x_nom_k||^2] = ||mu_k - x_nom_k||^2 + tr(Sigma_k)
        """
        fmean = tf.convert_to_tensor(fmean, dtype=default_float())
        Sdiag = tf.convert_to_tensor(fvar.diags, dtype=fmean.dtype)

        N = tf.shape(fmean)[0]
        P = tf.shape(fmean)[1]

        nominal = tf.convert_to_tensor(self.desired_nominal, dtype=fmean.dtype)

        # (N,P)
        diff = fmean - nominal
        sq = tf.reduce_sum(diff * diff, axis=1)          # (N,)

        # tr(Sigma_k) for full state: (N,)
        trS = tf.linalg.trace(Sdiag)                     # (N,)

        cost = self.tracking_weight * (sq + trS) / (self.sigma_nominal ** 2)  # (N,)
        return cost
    
    def _gauss_hermite_params(self, order: int):
        xi, wi = np.polynomial.hermite.hermgauss(order)  # for e^{-x^2}
        xi = tf.convert_to_tensor(xi, dtype=default_float())  # (K,)
        wi = tf.convert_to_tensor(wi, dtype=default_float())  # (K,)
        norm = tf.cast(SQRT_PI, dtype=default_float())
        return xi, wi / norm

    def _marginal_choleskies(self, fvar: Covariance, N: tf.Tensor) -> tf.Tensor:
        """
        Compute C_k such that S_kk = C_k C_k^T using diagonal blocks directly:
          S_kk = fvar.diags[k]
        Returns: C of shape (N,P,P)
        """
        # This assumes fvar.diags is (N,P,P).
        Sdiag = fvar.diags  # (N,P,P)
        Sdiag = 0.5 * (Sdiag + tf.transpose(Sdiag, perm=[0, 2, 1]))
        Sdiag = Sdiag + default_jitter() * tf.eye(tf.shape(Sdiag)[1], dtype=Sdiag.dtype)[None, :, :]
        return tf.linalg.cholesky(Sdiag)

    def _sample_posterior(self, fmean: tf.Tensor, fvar: Covariance, nsamples: int) -> tf.Tensor:
        """
        Draw samples using per-time marginals only (ignores cross-time correlations).

        Inputs:
          fmean: (N,P)
          fvar : Covariance with diags (N,P,P)
        Returns:
          samples: (S,N,P)
        """
        fmean = tf.convert_to_tensor(fmean)
        N = tf.shape(fmean)[0]
        P = tf.shape(fmean)[1]

        C = self._marginal_choleskies(fvar, N)  # (N,P,P)

        # z: (S,N,P)
        z = tf.random.normal([nsamples, N, P], dtype=fmean.dtype)

        # y_k = z_k @ C_k^T  (broadcast over samples)
        y = tf.einsum("snp,npq->snq", z, tf.transpose(C, perm=[0, 2, 1]))  # (S,N,P)

        return y + fmean[None, :, :]

    def variational_expectations(
        self,
        fmean: tf.Tensor,
        fvar: Covariance,
        method: str = "gauss_hermite",
        order: int = 10,
        nsamples: int = 8,
        Y=None,
    ):
        """
        Returns expected collision penalty per time index: shape (N,).
        Uses only marginal S_kk blocks from fvar.
        """
        fmean = tf.convert_to_tensor(fmean)
        N = tf.shape(fmean)[0]
        P = tf.shape(fmean)[1]

        if method == "gauss_hermite":
            xi, wi = self._gauss_hermite_params(order)  # (K,), (K,)
            X1, X2 = tf.meshgrid(xi, xi, indexing="ij")  # (K,K)
            W2 = wi[:, None] * wi[None, :]               # (K,K)

            # Z: (K,K,2)
            Z = tf.stack([X1, X2], axis=-1)

            # C: (N,P,P) from marginal S_kk
            C = self._marginal_choleskies(fvar, N)  # (N,P,P)

            # Use only position dims (dof)
            Cxy = C[:, :self.dof, :self.dof]  # (N,dof,dof)

            # f = mu + sqrt(2) * Cxy @ z
            # result: (N,K,K,dof)
            f = tf.einsum("nij,klj->nkli", Cxy, Z)
            f = fmean[:, None, None, :self.dof] + tf.cast(SQRT_TWO, fmean.dtype) * f

            # Signed distance to circle
            diff = f - self.center[None, None, None, :]          # (N,K,K,dof)
            sdist = tf.norm(diff, axis=-1) - self.radius         # (N,K,K)

            s = tf.cast(self.hinge_softness, fmean.dtype)
            penetration = tf.nn.softplus((self.epsilon - sdist) / s) * s  # (N,K,K)
            coll = (penetration / self.sigma_obs) ** 2                    # (N,K,K)

            E_coll = tf.reduce_sum(coll * W2[None, :, :], axis=(1, 2))      # (N,)
            E_track = self._expected_nominal_tracking_penalty(fmean, fvar)

            return - (E_coll + E_track)

        elif method == "monte_carlo":
            fsamples = self._sample_posterior(fmean, fvar, nsamples)  # (S,N,P)
            Pxy = fsamples[..., :self.dof]                            # (S,N,dof)

            diff = Pxy - self.center[None, None, :]                   # (S,N,dof)
            sdist = tf.norm(diff, axis=-1) - self.radius              # (S,N)

            s = tf.cast(self.hinge_softness, fmean.dtype)
            penetration = tf.nn.softplus((self.epsilon - sdist) / s) * s  # (S,N)
            coll = (penetration / self.sigma_obs) ** 2                    # (S,N)

            E_coll = tf.reduce_mean(coll, axis=0)                         # (N,)
            E_track = self._expected_nominal_tracking_penalty(fmean, fvar)

            return - (E_coll + E_track)

        else:
            raise NotImplementedError(
                "Expected log-likelihood calculation implemented only using Gauss-Hermite Quadrature or Monte Carlo"
            )

    def _log_prob(self, F, Y):
        raise NotImplementedError("PlanningLikelihood does not define pointwise log_prob.")

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError("PlanningLikelihood does not provide predictive log density.")

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError("PlanningLikelihood does not provide predictive moments.")
