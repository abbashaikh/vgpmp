from typing import List

import tensorflow as tf
from gpflow.config import default_float, default_jitter

from ..base import InputData
from ..dynamics.constant_velocity import ConstantVelocityModel
from ..utils.linear_algebra import BlockTriDiagCholesky, Cholesky, sym

class GaussMarkovKernel():
    """
    Input:
        X_data: shape (N,)
        anchor_vars: shape (P,)
    """
    def __init__(
        self,
        X_data: InputData, 
        dynamics: ConstantVelocityModel,
        anchor_vars: tf.Tensor,
    ):
        self.N = X_data.shape[0]
        self.P = dynamics.state_dimension

        self.dt = X_data[1:] - X_data[:-1]

        self.I = tf.eye(self.P, dtype=default_float())
        self.jitter_matrix = default_jitter() * self.I
        # initial and final state covariance
        self.anchor_vars = anchor_vars
        self.K0 = tf.linalg.diag(anchor_vars)
        self.KN = tf.linalg.diag(anchor_vars)
        # choleskies and inverse
        self.K0_chol = tf.linalg.cholesky(self.K0)
        self.K0_inv  = tf.linalg.cholesky_solve(self.K0_chol, self.I)
        # self.KN_chol = tf.linalg.cholesky(self.KN)
        # self.KN_inv  = tf.linalg.cholesky_solve(self.KN_chol, I_P)
        self.KN_inv = None

        ## Dynamics matrices
        self.Phi = dynamics.get_transition_matrices(self.dt)                                        # length N-1, each (P,P)
        self.Q = [Qi + self.jitter_matrix for Qi in dynamics.get_noise_matrices(self.dt)]    # length N-1, each (P,P) SPD
        self.u = dynamics.get_control_vectors(self.dt)                                              # length N-1, each (P,)
        self.Q_chol = [tf.linalg.cholesky(Qi) for Qi in self.Q]
        self.Qinv = [tf.linalg.cholesky_solve(self.Q_chol[i], self.I) for i in range(self.N - 1)]

        ## cholesky of Kinv = cholesky of (BinvT * Qinv * Binv)
        # efficient tri-diagonal only calculation
        self.Kinv_cholesky = self._build_Kinv_cholesky_blocks()
        #TODO: check for NaNs
        for i, t in enumerate(self.Kinv_cholesky.diags):
            tf.debugging.assert_all_finite(
                t, f"NaNs or Infs found in Kinv diags[{i}]"
            )

        for i, t in enumerate(self.Kinv_cholesky.sub_diags):
            tf.debugging.assert_all_finite(
                t, f"NaNs or Infs found in Kinv sub_diags[{i}]"
            )

        ## cholesky of K
        self.K_cholesky = self._build_covariance_from_precision(self.Kinv_cholesky)

        #TODO: check for NaNs
        for i, t in enumerate(self.K_cholesky.diags):
            tf.debugging.assert_all_finite(
                t, f"NaNs or Infs found in K diags[{i}]"
            )

        for i, t in enumerate(self.K_cholesky.off_diags):
            tf.debugging.assert_all_finite(
                t, f"NaNs or Infs found in K off_diags[{i}]"
            )


    def _build_Kinv_cholesky_blocks(self) -> BlockTriDiagCholesky:
        """
        Compute the block Cholesky L of the block-tridiagonal precision K^{-1} implied by (Phi, Q, K0, KN)
          diag[0]   = K0^{-1} + Phi_0^T Q_0^{-1} Phi_0
          off[0]    = - Phi_0^T Q_0^{-1}
          diag[i]   = Q_{i-1}^{-1} + Phi_i^T Q_i^{-1} Phi_i,    i=1..N-1
          off[i]    = - Phi_i^T Q_i^{-1},                       i=1..N-1
          diag[N]   = Q_{N-1}^{-1} + K_N^{-1}
        """
        def kdiag(i: int) -> tf.Tensor:
            """K^{-1}_{ii} block."""
            if i == 0:
                return self.K0_inv + tf.matmul(
                    self.Phi[0],
                    tf.matmul(self.Qinv[0], self.Phi[0]),
                    transpose_a=True
                )
            if i == self.N - 1:
                base = self.Qinv[-1]
                return base + (self.KN_inv if self.KN_inv is not None else tf.zeros_like(base))
            # 1..N-1
            return self.Qinv[i - 1] + tf.matmul(
                self.Phi[i],
                tf.matmul(self.Qinv[i], self.Phi[i]),
                transpose_a=True
            )

        def koff(i: int) -> tf.Tensor:
            """K^{-1}_{i,i-1} block, valid for i=1..N."""
            return -tf.matmul(self.Qinv[i-1], self.Phi[i-1])

        L_diags = []
        L_sub = []

        # i = 0
        K00 = kdiag(0)
        L00 = tf.linalg.cholesky(K00)
        tf.debugging.assert_all_finite(L00, f"L00 has NaN/Inf")
        L_diags.append(L00)

        # i = 1...N
        for i in range(1, self.N):
            # L_{i,i-1} = K^{-1}_{i,i-1} * (L_{i-1,i-1}^{-T})
            Kij = koff(i)
            Lij_transpose = tf.linalg.triangular_solve(
                L_diags[i - 1], tf.transpose(Kij), lower=True, adjoint=False
            )
            Lij = tf.transpose(Lij_transpose)
            L_sub.append(Lij)

            # S_i = K^{-1}_{ii} - L_{i,i-1} L_{i,i-1}^T
            Kii = sym(kdiag(i))
            tf.debugging.assert_all_finite(Kii, f"Kii has NaN/Inf at block {i}")

            Si = sym(Kii - tf.matmul(Lij, Lij, transpose_b=True))
            tf.debugging.assert_all_finite(Si,  f"Si has NaN/Inf at block {i}")

            # Si = Si + self.jitter_matrix

            # TODO: debugging
            Lii = tf.linalg.cholesky(Si)
            tf.debugging.assert_all_finite(Lii, f"Lii has NaN/Inf at block {i}")

            L_diags.append(Lii)

        return BlockTriDiagCholesky(diags=L_diags, sub_diags=L_sub)
    
    def _build_covariance_from_precision(self, Lp: BlockTriDiagCholesky):
        """
        Convert a block-bidiagonal precision Cholesky (J = L_p L_p^T) into the
        dense-block Cholesky of the covariance (K = L_K L_K^T) where L_K = L_p^{-1}.
        """
        N = Lp.num_diag_blocks
        P = Lp.block_size

        # rows[k] holds [L_K[k,0], ..., L_K[k,k]]
        rows: List[List[tf.Tensor]] = []

        for k in range(N):
            Lkk = Lp.diags[k]
            row_k: List[tf.Tensor] = []

            for j in range(k + 1):
                if j==k:
                    rhs = self.I
                else:
                    rhs = -tf.matmul(Lp.sub_diags[k - 1], rows[k - 1][j])

                # Solve L_p[k,k] * X = rhs  -> X = L_p[k,k]^{-1} rhs
                X_kj = tf.linalg.triangular_solve(Lkk, rhs, lower=True)
                row_k.append(X_kj)

            rows.append(row_k)

        L_diag_blocks: List[tf.Tensor] = [rows[i][i] for i in range(N)]
        L_off_blocks: List[tf.Tensor] = []
        for i in range(1, N):
            for j in range(i):
                L_off_blocks.append(rows[i][j])

        return Cholesky(diags=L_diag_blocks, off_diags=L_off_blocks)

    @property
    def block_dim(self) -> int:
        return self.P

    @property
    def num_data(self) -> int:
        return self.N