import tensorflow as tf
from gpflow.config import default_float

from ..base import MeanAndVariance, InputData
from ..dynamics import ConstantVelocityModel
from ..utils.linear_algebra import Cholesky

class GaussMarkovPosterior():
    """
    
    """
    def __init__(self, X_data: InputData, dynamics: ConstantVelocityModel):
        self.X_data = X_data
        self.dynamics = dynamics

    def predict_f(self, X_query: InputData, fmean: tf.Tensor, fvar: Cholesky, K_cholesky: Cholesky) -> MeanAndVariance:
        """
        Posterior mean and variance at query points
        Input:
            X_query: shape (M, 1)
            fmean: shape (N, P)
            fvar: of type Cholesky
            K_cholesky: of type Cholesky
        """
        mean_list = []
        variance_list = []

        for x in X_query:
            idx_next = tf.squeeze(tf.searchsorted(self.X_data, x, side='right'))
            idx_prev = idx_next - 1

            dt_prev = x - self.X_data[idx_prev]
            dt_next = self.X_data[idx_next] - x

            if dt_prev==0:
                mu = fmean[idx_prev, :]
                var = fvar.diags[idx_prev]
            elif dt_next==0:
                mu = fmean[idx_next, :]
                var = fvar.diags[idx_next]
            else:
                Phi_prev = self.dynamics.get_transition_matrices(dt_prev)[0]
                Phi_next = self.dynamics.get_transition_matrices(dt_next)[0]

                Q_prev = self.dynamics.get_noise_matrices(dt_prev)[0]
                Q_next_inv = self.dynamics.get_inverse_noise_matrices(dt_next)[0]

                # two terms of K_{fu} K_{uu}
                Lambda = tf.matmul(
                    Phi_next,
                    tf.matmul(Q_next_inv, Phi_next),
                    transpose_a=True,
                )
                L_prev = Phi_prev - tf.matmul(Q_prev, Lambda)           # shape (P,P)
                L_next = tf.matmul(
                    Q_prev,
                    tf.matmul(Phi_next, Q_next_inv, transpose_a=True)
                )                                                       # shape (P,P)
                L = tf.concat([L_prev, L_next], axis=1)                 # shape (P,2P)

                f_stack = tf.concat([
                    tf.transpose(tf.gather(fmean, idx_prev, axis=0)),   # shape (P,1)
                    tf.transpose(tf.gather(fmean, idx_next, axis=0)),   # shape (P,1)
                ], axis=0)
                mu = tf.matmul(L, f_stack)[:,0]                         # shape (P,)

                #TODO: K to Kinv cholesky
                K_i_ip1 = K_cholesky.get_prev_next_block(idx_prev)      # shape (2*P,2*P)
                S_i_ip1 = fvar.get_prev_next_block(idx_prev)            # shape (2*P,2*P)
                D = K_i_ip1 - S_i_ip1

                L = tf.concat([L_prev, L_next], axis=1)                 # shape (P,2*P)
                var = tf.matmul(
                    L,
                    tf.matmul(D, L, transpose_b=True)
                )

            mean_list.append(mu)
            variance_list.append(var)

        mean = tf.stack(mean_list, axis=0)                              # shape (M,P)
        variance = tf.stack(variance_list, axis=0)                      # shape (M,P,P)
        return mean, variance
    
    
    
