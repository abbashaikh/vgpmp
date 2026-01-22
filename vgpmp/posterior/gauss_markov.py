import tensorflow as tf
from gpflow.config import default_float

from ..base import MeanAndVariance, InputData
from ..dynamics import ConstantVelocityModel
from ..utils.linear_algebra import BlockTriDiagCovariance


class GaussMarkovPosterior:
    def __init__(self, X_data: InputData, dynamics: ConstantVelocityModel):
        self.X_data = X_data
        self.dynamics = dynamics

    def predict_f(self, X_query, fmean, fvar: BlockTriDiagCovariance, K_prior: BlockTriDiagCovariance):
        Xq = tf.reshape(tf.convert_to_tensor(X_query), [-1])      # (M,)
        Xd = tf.reshape(tf.convert_to_tensor(self.X_data), [-1])  # (N,)

        dtype = fmean.dtype
        Xq = tf.cast(Xq, Xd.dtype)

        M = tf.shape(Xq)[0]
        N = tf.shape(Xd)[0]
        P = tf.shape(fmean)[1]

        x0 = Xd[0]
        xN = Xd[N - 1]

        # Masks for boundary handling
        left_mask  = Xq <= x0
        right_mask = Xq >= xN
        mid_mask   = ~(left_mask | right_mask)

        # ---- interval indices for all points ----
        idx_next = tf.searchsorted(Xd, Xq, side="right")  # (M,)
        i = tf.clip_by_value(idx_next - 1, 0, N - 2)      # (M,)

        xi   = tf.gather(Xd, i)         # (M,)
        xip1 = tf.gather(Xd, i + 1)     # (M,)

        # Exact knot hits
        is_xi   = tf.equal(Xq, xi)
        is_xip1 = tf.equal(Xq, xip1)

        # dt's (broadcast-safe)
        dt_prev = tf.cast(Xq - xi, dtype)      # (M,)
        dt_next = tf.cast(xip1 - Xq, dtype)    # (M,)
        dt      = tf.cast(xip1 - xi, dtype)    # (M,)

        # ---- dynamics matrices ----
        Phi_prev = self.dynamics.get_transition_matrices(dt_prev)   # (M,P,P)
        Phi_next = self.dynamics.get_transition_matrices(dt_next)   # (M,P,P)
        Q_tau    = self.dynamics.get_noise_matrices(dt_prev)        # (M,P,P)
        Qinv_ip1 = self.dynamics.get_inverse_noise_matrices(dt)     # (M,P,P)

        Lambda = tf.matmul(Phi_next, tf.matmul(Qinv_ip1, Phi_next), transpose_a=True)  # (M,P,P)
        L_prev = Phi_prev - tf.matmul(Q_tau, Lambda)  # (M,P,P)
        L_next = tf.matmul(Q_tau, tf.matmul(Phi_next, Qinv_ip1, transpose_a=True))  # (M,P,P)
        L = tf.concat([L_prev, L_next], axis=2) # (M,P,2P)

        # ---- gather means for i and i+1 ----
        f_prev = tf.gather(fmean, i)         # (M,P)
        f_next = tf.gather(fmean, i + 1)     # (M,P)
        f_stack = tf.concat([f_prev, f_next], axis=1)   # (M,2P)
        f_stack = f_stack[..., None]                    # (M,2P,1)

        mu_mid = tf.matmul(L, f_stack)[:, :, 0]         # (M,P)

        # ---- gather covariance blocks ----
        K_i = K_prior.get_prev_next_block(i)      # (M,2P,2P)
        S_i = fvar.get_prev_next_block(i)         # (M,2P,2P)
        D   = K_i - S_i                           # (M,2P,2P)
        K_ii = K_prior.diag_block(i)              # (M,P,P)

        k_tau = tf.matmul(Phi_prev, tf.matmul(K_ii, Phi_prev, transpose_b=True)) + Q_tau  # (M,P,P)

        LD = tf.matmul(L, D)    # (M,P,2P)
        var_mid = k_tau - tf.matmul(LD, L, transpose_b=True)  # (M,P,P)

        # ---- boundary + exact-knot overrides ----
        # Start with mid results everywhere, then overwrite.
        mu  = mu_mid
        var = var_mid

        # Exact hits: xi -> use marginal at i, x_{i+1} -> marginal at i+1
        mu_xi   = tf.gather(fmean, i)         # (M,P)
        var_xi  = tf.gather(fvar.diags, i)    # (M,P,P)
        mu_xip1 = tf.gather(fmean, i + 1)
        var_xip1= tf.gather(fvar.diags, i + 1)

        mu  = tf.where(is_xi[:, None], mu_xi, mu)
        var = tf.where(is_xi[:, None, None], var_xi, var)

        mu  = tf.where(is_xip1[:, None], mu_xip1, mu)
        var = tf.where(is_xip1[:, None, None], var_xip1, var)

        # Left boundary -> index 0
        mu0  = tf.broadcast_to(fmean[0][None, :], [M, P])
        var0 = tf.broadcast_to(fvar.diags[0][None, :, :], [M, P, P])

        mu  = tf.where(left_mask[:, None], mu0, mu)
        var = tf.where(left_mask[:, None, None], var0, var)

        # Right boundary -> index N-1
        muN  = tf.broadcast_to(fmean[N-1][None, :], [M, P])
        varN = tf.broadcast_to(fvar.diags[N-1][None, :, :], [M, P, P])

        mu  = tf.where(right_mask[:, None], muN, mu)
        var = tf.where(right_mask[:, None, None], varN, var)

        return mu, var
