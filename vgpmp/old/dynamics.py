import tensorflow as tf

DTYPE = tf.float32

class ConstantVelocityModel():
    r"""
    The constant velocity LTV-SDE model is represented as:
    :math: `\dot{x}(t) = A(t)x(t) + u(t) + F(t)w(t)`
    where,
    :math: `A = [0 I; 0 0], \quad F = [0;I],`
    and :math: `w(t)` is white noise.
    """
    def __init__(self, dof: int, q_acc: float = 1.0):
        self.d = dof
        self.D = 2*self.d
        
        self.q_acc = q_acc

        self.Z = tf.zeros([self.d, self.d], dtype=DTYPE)
        self.I = tf.eye(self.d, dtype=DTYPE)

    def _system_matrices(self, times: tf.Tensor,):
        N = int(times.shape[0]) - 1

        A = tf.concat([
            tf.concat([self.Z, self.I], axis=1),
            tf.concat([self.Z, self.Z], axis=1)
        ], axis=0)                                          # shape (D, D)

        F = tf.concat([
            tf.zeros([self.d, self.d], dtype=DTYPE), self.I
        ], axis=0)                                          # shape (D, d)

        Qc = self.q_acc*tf.eye(self.d, dtype=DTYPE)               # shape (d, d)
        v = tf.zeros([self.D], DTYPE)                       # shape (D,)

        # replicate for each time-step
        A_list  = [A for _ in range(N)]
        F_list  = [F for _ in range(N)]
        Qc_list = [Qc for _ in range(N)]
        v_list  = [v for _ in range(N)]

        return A_list, F_list, Qc_list, v_list, self.D
    
    def transition_matrices(self, dt: tf.Tensor):
        Phis = []
        for k in range(dt.shape[0]):
            Phi_k = tf.concat(
                [
                    tf.concat([self.I, dt[k] * self.I], axis=1),
                    tf.concat([self.Z, self.I],         axis=1),
                ],
                axis=0,
            )                                               # shape (D, D)
            Phis.append(Phi_k)
        return Phis
    
    def noise_matrices(self, dt: tf.Tensor):
        Qs = []
        for k in range(dt.shape[0]):
            dt_k = dt[k]
            Qqq = (dt_k**3) / 3.0
            Qqv = (dt_k**2) / 2.0
            Qvv = dt_k

            Q_k = self.q_acc * tf.concat(
                [
                    tf.concat([Qqq * self.I, Qqv * self.I], axis=1),
                    tf.concat([Qqv * self.I, Qvv * self.I], axis=1),
                ],
                axis=0,
            )                                               # shape (D, D)
            Qs.append(Q_k)
        return Qs
    
    def control_vectors(self, dt: tf.Tensor):
        # considering zero initial controls
        return [tf.zeros([self.D], dtype=DTYPE) for _ in range(dt.shape[0])]
    
