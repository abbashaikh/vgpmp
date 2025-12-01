from __future__ import annotations
from typing import Tuple, Optional
import tensorflow as tf

DTYPE = tf.float32

class CollisionLikelihood():
    def __init__(
        self,
        N: int,
        D: int,
        obstacle_center: Tuple[float, float],
        obstacle_radius: float,
        grid_size: float = 10.0,
        epsilon: float = 0.3,
        sigma_obs: float = 0.05,
        sigma_box: float = 0.05,
        lambda_collision: float = 1.0,
        lambda_box: float = 1.0,
        enforce_box: bool = True,
    ):
        self.N = N
        self.D = D

        self.center = tf.convert_to_tensor(obstacle_center, dtype=DTYPE)  # shape [2]
        self.radius = tf.convert_to_tensor(obstacle_radius, dtype=DTYPE)

        self.grid_size = tf.convert_to_tensor(grid_size, dtype=DTYPE)
        self.epsilon = tf.convert_to_tensor(epsilon, dtype=DTYPE)
        self.sigma_obs = tf.convert_to_tensor(sigma_obs, dtype=DTYPE)
        self.sigma_box = tf.convert_to_tensor(sigma_box, dtype=DTYPE)
        self.lambda_collision = tf.convert_to_tensor(lambda_collision, dtype=DTYPE)
        self.lambda_box = tf.convert_to_tensor(lambda_box, dtype=DTYPE)
        self.enforce_box = enforce_box

    def _hinge(self, x: tf.Tensor) -> tf.Tensor:
        return tf.nn.relu(x)
    
    def _circle_signed_distance(self, Pxy: tf.Tensor) -> tf.Tensor:
        """
        Signed distance to circle boundary at each waypoint.
        Pxy: [S, N+1, 2] positions.
        Returns sdist: [S, N+1] where sdist > 0 outside the obstacle.
        """
        # ||p - c|| - R
        d = tf.norm(Pxy - self.center, axis=-1)  # [S, N+1]
        return d - self.radius
    
    def _box_violation(self, Pxy: tf.Tensor) -> tf.Tensor:
        """
        Soft penalty for leaving [0, L] × [0, L].
        Returns a non-negative violation per waypoint: [S, N+1].
        """
        x = Pxy[..., 0]  # [S, N+1]
        y = Pxy[..., 1]  # [S, N+1]

        lower_violation = self._hinge(-x) + self._hinge(-y)             # below 0
        upper_violation = self._hinge(x - self.grid_size) + self._hinge(y - self.grid_size)  # above L
        return lower_violation + upper_violation
    
    def log_likelihood(self, x_samples: tf.Tensor) -> tf.Tensor:
        """
        x_samples: [S, (N+1)*D] flattened samples
        Returns per-sample log-likelihood: [S]
        """
        S = tf.shape(x_samples)[0]
        X = tf.reshape(x_samples, [S, self.N+1, self.D])  # [S, N+1, D]

        # Use first two dimensions as (x, y)
        Pxy = tf.cast(X[..., :2], DTYPE)  # [S, N+1, 2]

        # --- Obstacle term (circular) ---
        sdist = self._circle_signed_distance(Pxy)  # [S, N+1]
        # Penalize if closer than (radius + epsilon)
        # penetration = max(0, (R + ε) - ||p-c||) = ReLU(ε - sdist)
        penetration = self._hinge(self.epsilon - sdist)  # [S, N+1]
        coll_cost = tf.reduce_sum((penetration / self.sigma_obs) ** 2, axis=-1)  # [S]

        # --- Box (workspace) term ---
        if self.enforce_box:
            v_box = self._box_violation(Pxy)  # [S, N+1]
            box_cost = tf.reduce_sum((v_box / self.sigma_box) ** 2, axis=-1)  # [S]
        else:
            box_cost = tf.zeros([S], dtype=DTYPE)

        total_cost = self.lambda_collision * coll_cost + self.lambda_box * box_cost  # [S]

        # Log-likelihood (up to an additive constant): -0.5 * total_cost
        return -0.5 * total_cost
    
