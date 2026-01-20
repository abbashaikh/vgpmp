import tensorflow as tf
from gpflow.config import default_float

class ConstantVelocityModel:
    r"""
    Constant velocity SDE:
        xdot = A x + F w
    with A = [[0, I],
              [0, 0]],
         F = [[0],
              [I]],
    and w(t) white noise on acceleration.
    """

    def __init__(self, dof: int, acceleration_noise: float = 1.0):
        self.dof = int(dof)
        self.P = 2 * self.dof
        self.acceleration_noise = tf.convert_to_tensor(acceleration_noise, dtype=default_float())

        # Convenience (2D) blocks
        self.Z = tf.zeros([self.dof, self.dof], dtype=default_float())
        self.I = tf.eye(self.dof, dtype=default_float())

    @property
    def state_dimension(self):
        return self.P

    def get_transition_matrices(self, dt: tf.Tensor) -> tf.Tensor:
        """
        dt: (B,) or scalar
        returns: (B, P, P) or (P,P) if scalar input
        """
        dt = tf.convert_to_tensor(dt, dtype=default_float())
        squeeze_back = (dt.shape.rank == 0)

        dt = tf.reshape(dt, [-1])  # (B,)
        B = tf.shape(dt)[0]

        I = self.I[None, :, :]  # (1,dof,dof)
        Z = self.Z[None, :, :]  # (1,dof,dof)

        dtI = dt[:, None, None] * I  # (B,dof,dof)

        top = tf.concat([tf.broadcast_to(I, [B, self.dof, self.dof]), dtI], axis=2)  # (B,dof,2dof)
        bot = tf.concat([tf.broadcast_to(Z, [B, self.dof, self.dof]),
                         tf.broadcast_to(I, [B, self.dof, self.dof])], axis=2)       # (B,dof,2dof)
        Phi = tf.concat([top, bot], axis=1)  # (B,2dof,2dof) = (B,P,P)

        if squeeze_back:
            return Phi[0]
        return Phi

    def get_noise_matrices(self, dt: tf.Tensor) -> tf.Tensor:
        """
        Discrete-time process noise covariance for white-noise acceleration.
        dt: (B,) or scalar
        returns: (B,P,P) or (P,P) if scalar input
        """
        dt = tf.convert_to_tensor(dt, dtype=default_float())
        squeeze_back = (dt.shape.rank == 0)

        dt = tf.reshape(dt, [-1])  # (B,)
        B = tf.shape(dt)[0]

        Qqq = (dt ** 3) / tf.cast(3.0, dt.dtype)   # (B,)
        Qqv = (dt ** 2) / tf.cast(2.0, dt.dtype)   # (B,)
        Qvv = dt                                    # (B,)

        I = self.I[None, :, :]  # (1,dof,dof)
        I = tf.broadcast_to(I, [B, self.dof, self.dof])

        QqqI = Qqq[:, None, None] * I
        QqvI = Qqv[:, None, None] * I
        QvvI = Qvv[:, None, None] * I

        top = tf.concat([QqqI, QqvI], axis=2)  # (B,dof,2dof)
        bot = tf.concat([QqvI, QvvI], axis=2)  # (B,dof,2dof)
        Q = tf.concat([top, bot], axis=1)      # (B,2dof,2dof)

        Q = self.acceleration_noise * Q

        if squeeze_back:
            return Q[0]
        return Q

    def get_inverse_noise_matrices(self, dt: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
        """
        Inverse of the discrete-time process noise covariance (for dt>0).
        dt: (B,) or scalar
        returns: (B,P,P) or (P,P) if scalar input
        """
        dt = tf.convert_to_tensor(dt, dtype=default_float())
        squeeze_back = (dt.shape.rank == 0)

        dt = tf.reshape(dt, [-1])  # (B,)
        B = tf.shape(dt)[0]

        dt_safe = tf.maximum(dt, tf.cast(eps, dt.dtype))

        Qqq = tf.cast(12.0, dt.dtype) / (dt_safe ** 3)   # (B,)
        Qqv = -tf.cast(6.0, dt.dtype) / (dt_safe ** 2)   # (B,)
        Qvv = tf.cast(4.0, dt.dtype) / dt_safe           # (B,)

        I = self.I[None, :, :]
        I = tf.broadcast_to(I, [B, self.dof, self.dof])

        QqqI = Qqq[:, None, None] * I
        QqvI = Qqv[:, None, None] * I
        QvvI = Qvv[:, None, None] * I

        top = tf.concat([QqqI, QqvI], axis=2)
        bot = tf.concat([QqvI, QvvI], axis=2)
        Qinv = tf.concat([top, bot], axis=1)  # (B,P,P)

        Qinv = (tf.cast(1.0, dt.dtype) / self.acceleration_noise) * Qinv

        if squeeze_back:
            return Qinv[0]
        return Qinv

    def get_control_vectors(self, dt: tf.Tensor) -> tf.Tensor:
        """
        returns: (B,P)
        """
        dt = tf.convert_to_tensor(dt, dtype=default_float())
        dt = tf.reshape(dt, [-1])
        B = tf.shape(dt)[0]
        return tf.zeros([B, self.P], dtype=default_float())

    def initate_traj(self, times, start, goal, method: str = ""):
        start = tf.convert_to_tensor(start, default_float())        # shape (P,)
        goal = tf.convert_to_tensor(goal, default_float())
        times = tf.convert_to_tensor(times, default_float())        # shape (N,1)

        start_pos = tf.expand_dims(start[:self.dof], axis=0)        # (1,dof)
        goal_pos  = tf.expand_dims(goal[:self.dof],  axis=0)        # (1,dof)

        T = times[-1] - times[0]
        w = (times - times[0]) / T                                  # shape (N,1)

        positions = start_pos + w * (goal_pos - start_pos)          # shape (N,dof)

        v_mid = (positions[2:] - positions[:-2]) / (times[2:] - times[:-2])  # (N-2,dof)
        zeros = tf.zeros((1, self.dof), dtype=default_float())
        velocities = tf.concat([zeros, v_mid, zeros], axis=0)                  # (N,dof)

        return tf.concat([positions, velocities], axis=1)                      # (N,P)
