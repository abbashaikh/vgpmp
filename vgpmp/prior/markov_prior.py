import tensorflow as tf
import gpflow as gf

from ..dynamics.constant_velocity import ConstantVelocityModel
# from .block_tridiag import BlockTriDiag

DTYPE = tf.float32


class GaussMarkovPrior(gf.Module):
    """
    Shapes:
      - times: (N+1,)
      - K0, mu0: (D,D), (D,)
      - KN, muN: (D,D), (D,)
    """
    def __init__(
        self,
        times: tf.Tensor,
        dynamics: ConstantVelocityModel,
        start: tuple[tf.Tensor, tf.Tensor],
        end: tuple[tf.Tensor, tf.Tensor],
        jitter: float = 1e-6,
        name: str = "gauss_markov_prior",
    ):
        super().__init__(name=name)

        self.N = int(times.shape[0]) - 1    # number of segments
        self.times = times
        self.dt = self.times[1:] - self.times[:-1]
        
        ## Initial and goal stats
        # shape K_i: (D, D); shape mu_i: (D,)
        K0, mu0 = start
        self.D = mu0.shape[0]
        # create jitter matrix
        self.jitter = tf.cast(jitter, DTYPE)
        I_D = tf.eye(self.D, dtype=DTYPE)
        # add jitter (for stability)
        self.K0 = tf.cast(K0, DTYPE) + self.jitter * I_D
        self.mu0 = tf.reshape(tf.cast(mu0, DTYPE), [self.D])
        KN, muN = end
        self.KN = tf.cast(KN, DTYPE) + self.jitter * I_D
        self.muN = tf.reshape(tf.cast(muN, DTYPE), [self.D])
        # choleskies and inverse
        self.K0_chol = tf.linalg.cholesky(self.K0)
        self.K0_inv  = tf.linalg.cholesky_solve(self.K0_chol, I_D)
        self.KN_chol = tf.linalg.cholesky(self.KN)
        self.KN_inv  = tf.linalg.cholesky_solve(self.KN_chol, I_D)

        ## Dynamics matrices
        self.Phi = dynamics.transition_matrices(self.dt)                                        # length N, each (D,D)
        self.Q = [Qi + self.jitter * I_D for Qi in dynamics.noise_matrices(self.dt)]            # length N, each (D,D) SPD
        self.u = dynamics.control_vectors(self.dt)                                              # length N, each (D,)
        # choleskies and inverse
        self.Q_chol  = [tf.linalg.cholesky(Qi) for Qi in self.Q]
        self.Q_inv   = [tf.linalg.cholesky_solve(self.Q_chol[i], I_D) for i in range(self.N)]

        ## Prior mean
        # mu(i+1) = Phi(i)*mu(i) + u(i)
        # Linear interpolation between start and goal means across the support grid
        dof = self.D // 2
        alphas = tf.reshape(tf.linspace(0.0, 1.0, self.N + 1), [self.N + 1, 1])
        p0 = tf.reshape(self.mu0[:dof], [1, dof])
        pN = tf.reshape(self.muN[:dof], [1, dof])
        mu_p = tf.cast((1.0 - alphas) * p0 + alphas * pN, DTYPE)  # shape (N+1, D)
        # Velocities: zero at endpoints, finite-difference in the middle
        mu_v = tf.zeros_like(mu_p)
        dt_mid = tf.reshape(self.times[2:] - self.times[:-2], [-1, 1])
        mid_v = (mu_p[2:] - mu_p[:-2]) / dt_mid  # central diff
        mu_v = tf.concat(
            [
                mu_v[:1],                # zero at start
                mid_v,                   # middle velocities
                mu_v[-1:],               # zero at end
            ],
            axis=0,
        )
        self.mu = tf.concat([mu_p, mu_v], axis=-1)  # shape (N+1, D)
        # mus = [self.mu0]
        # for i in range(self.N):
        #     mus.append(tf.linalg.matvec(self.Phi[i], mus[-1]) + self.u[i])
        # self.mu = tf.concat([tf.reshape(m, [1, self.D]) for m in mus], axis=0)  # shape (N+1, D)

        ## Kinv = BinvT * Qinv * Binv
        # efficient tri-diagonal only calculation
        self.Kinv = self._build_Kinv_blocks()

    def _build_Kinv_blocks(self) -> BlockTriDiag:
        """
        Construct explicit block-tridiagonal Kinv from Phi, Q, K0, KN:
          diag[0]   = K0^{-1} + Phi_0^T Q_0^{-1} Phi_0
          off[0]    = - Phi_0^T Q_0^{-1}
          diag[i]   = Q_{i-1}^{-1} + Phi_i^T Q_i^{-1} Phi_i,    i=1..N-1
          off[i]    = - Phi_i^T Q_i^{-1},                       i=1..N-1
          diag[N]   = Q_{N-1}^{-1} + K_N^{-1}
        """
        diag = []
        off = []

        # First block
        term0 = tf.matmul(tf.transpose(self.Phi[0]), tf.matmul(self.Q_inv[0], self.Phi[0]))
        diag.append(self.K0_inv + term0)
        off.append(-tf.matmul(tf.transpose(self.Phi[0]), self.Q_inv[0]))

        # Middle blocks
        for i in range(1, self.N):
            term_i = tf.matmul(tf.transpose(self.Phi[i]), tf.matmul(self.Q_inv[i], self.Phi[i]))
            di = self.Q_inv[i - 1] + term_i
            diag.append(di)
            off.append(-tf.matmul(tf.transpose(self.Phi[i]), self.Q_inv[i]))

        # Last block
        last = self.Q_inv[self.N - 1]
        diag.append(last)

        # off has length N; diag has length N+1
        return BlockTriDiag(diag=diag, off=off, jitter=self.jitter)

    @property
    def block_dim(self) -> int:
        return self.D

    @property
    def num_segments(self) -> int:
        return self.N

    @property
    def mean(self) -> tf.Tensor:
        return self.mu
    
    @property
    def precision(self) -> BlockTriDiag:
        return self.Kinv