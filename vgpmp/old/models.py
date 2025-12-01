from typing import List
import tensorflow as tf
import gpflow

from .prior import GaussMarkovPrior
from .likelihood import CollisionLikelihood
from .matrix_utils import logdet_gauss_markov_cov

DTYPE = tf.float32
MIN_SCALE = tf.cast(1e-3, DTYPE)  # lower bound for diag entries

class VGP(gpflow.models.BayesianModel):
    def __init__(self, prior: GaussMarkovPrior, likelihood: CollisionLikelihood):
        r"""
        Covariance of the approximate posterior :math: `q(x)`
        is parameterized as :math: `S = \tilde{A} \tilde{Q} \tilde{A}^T`
        where :math: `\tilde{A}` is dense lower triangular
        and :math: `\tilde{Q}` is block diagonal.
        """
        self.prior = prior
        # Prior mean
        self.mu = prior.mean
        # Prior covariance
        self.Kinv = prior.precision
        # cache cholesky and logdet of Kinv
        self.Kinv.build_cache()
        self._logdet_K_cache = -self.Kinv.logdet(cache_only=True)

        self.N = prior.num_segments
        self.D = prior.block_dim

        # Approximate posterior mean
        self.m = gpflow.Parameter(tf.zeros([(self.N+1), self.D], dtype=DTYPE))
        # Approximate posterior covariance
        self.A_tilde = gpflow.Parameter(
            tf.eye((self.N+1)*self.D, dtype=DTYPE),
            transform=gpflow.utilities.triangular()      # positive diag with floor
        )
        # blocks of C, where Q = C*C^T
        self.C_blocks: List[gpflow.Parameter] = [
            gpflow.Parameter(
                tf.eye(self.D, dtype=DTYPE),
                transform=gpflow.utilities.triangular()  # positive diag with floor
            )
            for _ in range(self.N+1)
        ]
        
        self.likelihood = likelihood

    def _sample_q(self, nsamples: int):
        r"""
        To get samples: 
        :math: `x \sim \mathcal{N}(m, S)`,
        First sample :math: `z_i \sim \mathcal{N}(0, I)`,
        Then sample :math: `y_i \sim \mathcal{N}(0, \tilde{Q}_i)`.
        Concatenate all :math: `y_i` samples into :math: `y`.
        And finally get required :math: `x` samples.
        """
        # sample z_i from a standard normal
        z_blocks = [tf.random.normal([nsamples, self.D], dtype=DTYPE) for _ in range(self.N + 1)]
        # y_i ~ normal(0, Q_tilde_i)
        y_blocks = [tf.matmul(zi, Ci, transpose_b=True) for zi, Ci in zip(z_blocks, self.C_blocks)]
        y_samples = tf.concat(y_blocks, axis=-1)
        # x ~ normal(0, S)
        # TODO: this step has computational complexity O((N+1)^2*D^2)
        x_samples = tf.matmul(y_samples, self.A_tilde, transpose_b=True)
        # x ~ normal(m, S)
        x_samples = x_samples + tf.reshape(self.m, [1, (self.N + 1) * self.D])
        return x_samples    # shape (nsamples, (N+1)*D)
    
    def _get_block(self, R: tf.Tensor, p: int, q: int) -> tf.Tensor:
        """
        Return the (p, q) block of size [D, D] from the dense R
        """
        r0, r1 = p * self.D, (p + 1) * self.D
        c0, c1 = q * self.D, (q + 1) * self.D
        return R[r0:r1, c0:c1]
    
    def _trace_KinvS(self): # TODO: complexity O((N+1)^2*D^3)
        """
        Computes tr(Kinv * S) with S = A_tilde * blkdiag(Qi_tilde) * A_tilde^T and Qi_tilde = C_i C_i^T,
        without forming S or Q_tilde, using only block operations.

        ------
        Kinv        : BlockTriDiag (symmetric), with N+1 diagonal blocks and N off blocks.
        A_tilde     : [n, n] dense lower-triangular
        C_blocks    : list of k tensors, each [D, D]
        ------

        Output
        ------
        scalar tf.Tensor with tr(Kinv*S)
        """
        tr_sum = tf.constant(0.0, dtype=DTYPE)
        for i in range(self.N + 1):
            Y_pi = [None] * (self.N + 1)
            for p in range(self.N + 1):
                acc = tf.zeros([self.D, self.D], dtype=DTYPE)

                if p >= 1:
                    q = p - 1
                    if q >= i:  # only for lower-triangular terms
                        Kinv_pq = self.Kinv.off[p - 1]                    # (p, p-1)
                        A_qi = self._get_block(self.A_tilde, q, i)
                        acc = acc + tf.matmul(Kinv_pq, A_qi)

                # q = p term (diagonal)
                if p >= i:
                    Kinv_pp = self.Kinv.diag[p]
                    A_pi = self._get_block(self.A_tilde, p, i)
                    acc = acc + tf.matmul(Kinv_pp, A_pi)

                # q = p+1 term (upper off-diagonal = transpose of lower)
                if p + 1 < self.N + 1:
                    q = p + 1
                    if q >= i:
                        Kinv_pq = tf.linalg.adjoint(self.Kinv.off[p])      # (p, p+1) = off[p]^T
                        A_qi = self._get_block(self.A_tilde, q, i)
                        acc = acc + tf.matmul(Kinv_pq, A_qi)

                Y_pi[p] = acc

            # B_ii = sum_{p=i..k-1} A_{p,i}^T * Y_{p,i}
            Bii = tf.zeros([self.D, self.D], dtype=DTYPE)
            for p in range(i, self.N + 1):  # lower-triangular R => R_{p,i} nonzero only for p >= i
                A_pi = self._get_block(self.A_tilde, p, i)
                Bii = Bii + tf.matmul(A_pi, Y_pi[p], transpose_a=True)

            # Accumulate tr(B_ii * W_i) via Cholesky: tr(B_ii * L_i L_i^T) = tr(L_i^T B_ii L_i)
            Ci = self.C_blocks[i]                             # [D, D]
            T  = tf.matmul(Bii, Ci)                           # [D, D]
            tr_sum = tr_sum + (
                tf.reduce_sum(
                    tf.linalg.diag_part(
                        tf.matmul(Ci, T, transpose_a=True)
                    )
                )
            )

        return tr_sum
    
    def elbo(self, nsamples: int):
        x_samples = self._sample_q(nsamples)
        ell = tf.reduce_mean(self.likelihood.log_likelihood(x_samples))

        r = self.mu - self.m
        kl = 0.5*(
            self._trace_KinvS()
            + self.Kinv.quad_form(tf.reshape(r, [-1]))
            - logdet_gauss_markov_cov(self.A_tilde, self.C_blocks)
            + self._logdet_K_cache
        )
        return ell - kl

    def maximum_log_likelihood_objective(self):
        return self.elbo()
    
    def objective_closure(self, nsamples):
        return -self.elbo(nsamples=nsamples)