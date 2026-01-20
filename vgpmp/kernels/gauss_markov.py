from typing import List

import tensorflow as tf
from gpflow.config import default_float, default_jitter

from ..base import InputData
from ..dynamics.constant_velocity import ConstantVelocityModel
from ..utils.linear_algebra import BlockTriDiagMatrix, BlockTriDiagCovariance, sym


class GaussMarkovKernel:
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

        # inverses for boundary terms in precision
        self.K0_chol = tf.linalg.cholesky(self.K0)
        self.K0_inv = tf.linalg.cholesky_solve(self.K0_chol, self.I)
        self.KN_inv = None

        ## Dynamics matrices
        # TODO: get_matrices now return tensors and not lists
        self.Phi = dynamics.get_transition_matrices(self.dt)  # length N-1, each (P,P)
        self.Q = [Qi for Qi in dynamics.get_noise_matrices(self.dt)]  # length N-1
        self.Qinv = [Qinv_i for Qinv_i in dynamics.get_inverse_noise_matrices(self.dt)]  # length N-1
        self.u = dynamics.get_control_vectors(self.dt)  # length N-1, each (P,)

        # Precision matrix blocks
        self.Kinv = self._build_Kinv_blocks()

        # sanity checks
        for i in tf.range(self.Kinv.num_diag_blocks):
            tf.debugging.assert_all_finite(self.Kinv.diags[i], f"NaNs or Infs found in Kinv.diags[{i}]")
        for i in tf.range(self.Kinv.num_diag_blocks - 1):
            tf.debugging.assert_all_finite(self.Kinv.sub_diags[i], f"NaNs or Infs found in Kinv.sub_diags[{i}]")

        # Covariance matrix blocks
        self.K = self._build_covariance_from_precision(self.Kinv)

        # # sanity checks
        for i in tf.range(self.K.N):
            tf.debugging.assert_all_finite(self.K.diags[i], f"NaNs or Infs found in K.diags[{i}]")
        for i in tf.range(tf.shape(self.K.sub_diags)[0]):
            tf.debugging.assert_all_finite(self.K.sub_diags[i], f"NaNs or Infs found in K.off_diags[{i}]")

    @property
    def block_dim(self) -> int:
        return self.P

    @property
    def num_data(self) -> int:
        return self.N
    
    def _build_Kinv_blocks(self) -> BlockTriDiagMatrix:
        """
        Build the block-tridiagonal precision matrix J = K^{-1} implied by (Phi, Q, K0, KN):

          J_00     = K0^{-1} + Phi_0^T Q_0^{-1} Phi_0
          J_i,i    = Q_{i-1}^{-1} + Phi_i^T Q_i^{-1} Phi_i,    i=1..N-2
          J_{N-1}  = Q_{N-2}^{-1} + KN^{-1}  (KN^{-1} optional)

          J_{i,i-1} = - Q_{i-1}^{-1} Phi_{i-1}
          J_{i-1,i} = J_{i,i-1}^T,  i=1..N-1
        """
        N = tf.convert_to_tensor(self.N)

        def jdiag(i: tf.Tensor) -> tf.Tensor:
            i = tf.convert_to_tensor(i)

            def case0():
                Phi0 = self.Phi[0]
                Q0i = self.Qinv[0]
                return self.K0_inv + tf.matmul(Phi0, tf.matmul(Q0i, Phi0), transpose_a=True)

            def caselast():
                base = self.Qinv[-1]  # Q_{N-2}^{-1}
                KN = self.KN_inv if (self.KN_inv is not None) else tf.zeros_like(base)
                return base + KN

            def casemid():
                # i in 1..N-2
                Q_im1 = tf.gather(self.Qinv, i - 1)
                Phi_i = tf.gather(self.Phi, i)
                Q_i = tf.gather(self.Qinv, i)
                return Q_im1 + tf.matmul(Phi_i, tf.matmul(Q_i, Phi_i), transpose_a=True)

            return tf.cond(
                tf.equal(i, 0),
                case0,
                lambda: tf.cond(tf.equal(i, N - 1), caselast, casemid),
            )

        def jsub(i: tf.Tensor) -> tf.Tensor:
            """
            Return J_{i, i-1} for i=1..N-1, stored as sub_diags[i-1].
            """
            Qinv_im1 = tf.gather(self.Qinv, i - 1)
            Phi_im1 = tf.gather(self.Phi, i - 1)
            return -tf.matmul(Qinv_im1, Phi_im1)

        diags_ta = tf.TensorArray(dtype=self.K0_inv.dtype, size=N)
        sub_ta = tf.TensorArray(dtype=self.K0_inv.dtype, size=N - 1)

        # diagonal blocks
        for i in tf.range(N):
            Jii = sym(jdiag(i)) + self.jitter_matrix
            tf.debugging.assert_all_finite(Jii, "Jii has NaN/Inf")
            diags_ta = diags_ta.write(i, Jii)

        # sub-diagonal blocks
        for i in tf.range(1, N):
            Jij = jsub(i)
            tf.debugging.assert_all_finite(Jij, "J_{i,i-1} has NaN/Inf")
            sub_ta = sub_ta.write(i - 1, Jij)

        return BlockTriDiagMatrix(diags=diags_ta.stack(), sub_diags=sub_ta.stack())

    def _build_covariance_from_precision(self, J: BlockTriDiagMatrix) -> "BlockTriDiagCovariance":
        """
        Convert precision J (block tri-diagonal) into covariance K = J^{-1}, returning
        ONLY the block-tridiagonal part of K:
          - diags:     K_{i,i}
          - sub_diags: K_{i+1,i}
        All other off-diagonal blocks are ignored.
        """
        # TODO: alternative to densify
        N = J.num_diag_blocks
        P = J.block_size
        dtype = J.diags.dtype

        # Dense precision
        J_dense = J.to_dense()
        J_dense = sym(J_dense) + default_jitter() * tf.eye(tf.shape(J_dense)[0], dtype=dtype)

        # Dense covariance via Cholesky solve
        L = tf.linalg.cholesky(J_dense)
        I_big = tf.eye(tf.shape(J_dense)[0], dtype=dtype)
        K_dense = tf.linalg.cholesky_solve(L, I_big)  # (N*P, N*P)

        # Reshape to block view: (N, P, N, P)
        K4 = tf.reshape(K_dense, (N, P, N, P))

        # Diagonal blocks K_{i,i}: (N,P,P)
        # K4[i,:,i,:] -> (P,P)
        diags = tf.transpose(tf.linalg.diag_part(K4), perm=[0, 2, 1])  # (N,P,P)

        # Sub-diagonal blocks K_{i+1,i}: (N-1,P,P)
        # K4[i+1,:,i,:]
        sub_ta = tf.TensorArray(dtype=dtype, size=N - 1)
        for i in tf.range(N - 1):
            K_ip1_i = K4[i + 1, :, i, :]  # (P,P)
            sub_ta = sub_ta.write(i, K_ip1_i)
        sub_diags = sub_ta.stack()

        return BlockTriDiagCovariance(diags=diags, sub_diags=sub_diags)