import tensorflow as tf
from gpflow.config import default_float

def softplus_inverse(y: tf.Tensor) -> tf.Tensor:
    """
    Numerically stable inverse of softplus for y>0:
        s^{-1}(y) = log(exp(y) - 1)
    For large y this approximates y, so clamp the branch to avoid overflow.
    """
    y = tf.convert_to_tensor(y, dtype=default_float())
    large = tf.math.log(tf.float64.max) if y.dtype == tf.float64 else tf.math.log(tf.float32.max)
    return tf.where(
        y > 20.0,
        y,
        tf.math.log(tf.math.expm1(tf.minimum(y, large))),
    )