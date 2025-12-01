from typing import List
import tensorflow as tf

def logdet_gauss_markov_cov(A: tf.Tensor, L_blocks: List[tf.Tensor]):
    """
    Log-determinant of matrices in a form similar to
    covariance matrix of the Gauss-Markov model
    """
    tr1 = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(A)), axis=-1)
    tr2 = tf.add_n([
        2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Li)), axis=-1)
        for Li in L_blocks
    ])
    return tr1 + tr2


