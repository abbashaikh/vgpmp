import tensorflow as tf
from gpflow.config import default_float

class ConstantVelocityModel():
    r"""
    The constant velocity LTV-SDE model is represented as:
    :math: `\dot{x}(t) = A(t)x(t) + u(t) + F(t)w(t)`
    where,
    :math: `A = [0 I; 0 0], \quad F = [0;I],`
    and :math: `w(t)` is white noise.
    """
    def __init__(self, dof: int, acceleration_noise: float = 1.0):
        self.dof = dof
        self.P = 2*self.dof 
        
        self.acceleration_noise = acceleration_noise

        self.Z = tf.zeros([self.dof, self.dof], dtype=default_float())
        self.I = tf.eye(self.dof, dtype=default_float())

    @property
    def state_dimension(self):
        return self.P
    
    @property
    def num_times(self):
        return self.N
    
    def _system_matrices(self, times: tf.Tensor):
        N = int(times.shape[0])
        A = tf.concat([
            tf.concat([self.Z, self.I], axis=1),
            tf.concat([self.Z, self.Z], axis=1)
        ], axis=0)                                                      # shape (P, P)

        F = tf.concat([
            tf.zeros([self.dof, self.dof], dtype=default_float()), self.I
        ], axis=0)                                                      # shape (P, dof)

        Qc = self.q_acc*tf.eye(self.dof, dtype=default_float())         # shape (dof, dof)
        v = tf.zeros([self.D], default_float())                         # shape (P,)

        # replicate for each time-step
        A_list  = [A for _ in range(self.N)]
        F_list  = [F for _ in range(self.N)]
        Qc_list = [Qc for _ in range(self.N)]
        v_list  = [v for _ in range(self.N)]

        return A_list, F_list, Qc_list, v_list, self.D
    
    def _transition_matrix(self, dt: tf.Tensor):
        Phi = tf.concat(
            [
                tf.concat([self.I, dt * self.I], axis=1),
                tf.concat([self.Z, self.I], axis=1),
            ],
            axis=0,
        ) 
        return Phi
    
    def _noise_matrix(self, dt):
        Qqq = (dt**3) / 3.0
        Qqv = (dt**2) / 2.0
        Qvv = dt

        Q = self.acceleration_noise * tf.concat(
            [
                tf.concat([Qqq * self.I, Qqv * self.I], axis=1),
                tf.concat([Qqv * self.I, Qvv * self.I], axis=1),
            ],
            axis=0,
        )
        return Q
    
    # TODO
    def _inverse_noise_matrix(self, dt):
        #TODO: handle dt=0.0
        Qqq = 12.0 * (1 / dt**3)
        Qqv = -6.0 * (1 / dt**2)
        Qvv = 4 * (1 / dt)

        Q = (1 / self.acceleration_noise) * tf.concat(
            [
                tf.concat([Qqq * self.I, Qqv * self.I], axis=1),
                tf.concat([Qqv * self.I, Qvv * self.I], axis=1),
            ],
            axis=0,
        )
        return Q 
    
    def get_transition_matrices(self, dt: tf.Tensor):
        Phis = []
        for k in range(dt.shape[0]):
            Phi_k = self._transition_matrix(dt[k])           # shape (P, P)
            Phis.append(Phi_k)
        return Phis
    
    def get_noise_matrices(self, dt: tf.Tensor):
        Qs = []
        for k in range(dt.shape[0]):
            Q_k = self._noise_matrix(dt[k])                  # shape (P, P)
            Qs.append(Q_k)
        return Qs
    
    def get_inverse_noise_matrices(self, dt: tf.Tensor):
        Qs_inv = []
        for k in range(dt.shape[0]):
            Q_k_inv = self._noise_matrix(dt[k])             # shape (P, P)
            Qs_inv.append(Q_k_inv)
        return Qs_inv
    
    def get_control_vectors(self, dt: tf.Tensor):
        # considering zero initial controls
        return [tf.zeros([self.P], dtype=default_float()) for _ in range(dt.shape[0])]
    
    def initate_traj(self, times, start, goal, method: str = ""):
        start = tf.convert_to_tensor(start, default_float())        # shape (P,)
        goal = tf.convert_to_tensor(goal, default_float())
        times = tf.convert_to_tensor(times, default_float())        # shape (N,1)

        start_pos = tf.expand_dims(start[:self.dof], axis=0)        # (1,dof)
        goal_pos  = tf.expand_dims(goal[:self.dof],  axis=0)        # (1,dof)

        T = times[-1] - times[0]
        w = (times - times[0]) / T                                  # shape (N,1)

        positions = start_pos + w * (goal_pos - start_pos)          # shape (N,dof)    

        v_mid = (positions[2:] - positions[:-2]) / (times[2:] - times[:-2])       # shape (N-2,dof)
        zeros = tf.zeros((1,self.dof), dtype=default_float())
        velocities = tf.concat([zeros, v_mid, zeros], axis=0)                     # shape (N,dof)

        return tf.concat([positions, velocities], axis=1)                         # shape (N,P)

    
