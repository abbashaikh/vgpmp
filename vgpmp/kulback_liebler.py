import tensorflow as tf
from gpflow.config import default_float, default_jitter

from .utils.linear_algebra import BlockTriDiagCovariance, BlockTriDiagMatrix, sym


def calc_trace_J_S(J: BlockTriDiagMatrix, S: BlockTriDiagCovariance) -> tf.Tensor:
    """
    Compute tr(J S) where
      - J is block tri-diagonal
      - S is block-symmetric covariance

    Uses:
      tr(JS) = sum_i tr(J_ii S_ii) + 2 * sum_{i=0}^{N-2} tr(J_{i,i+1} S_{i,i+1})
    because J_{i+1,i} = J_{i,i+1}^T and S_{i+1,i} = S_{i,i+1}^T.
    """
    dtype = J.diags.dtype
    N = tf.shape(J.diags)[0]

    # Diagonal part: sum_i tr(J_ii S_ii) = sum_i sum_{a,b} J_ii[a,b] * S_ii[a,b]
    diag_term = tf.reduce_sum(J.diags * S.diags)

    # Off part: 2 * sum_i tr(J_{i,i+1} S_{i,i+1})
    # with J_{i,i+1} = Jsub[i]^T, S_{i,i+1} = Ssub[i]^T
    Jsup = tf.transpose(J.sub_diags, perm=[0, 2, 1])  # (N-1,P,P)
    Ssup = tf.transpose(S.sub_diags, perm=[0, 2, 1])  # (N-1,P,P)
    off_term = tf.reduce_sum(Jsup * Ssup)

    two = tf.cast(2.0, dtype)
    return tf.cast(diag_term, dtype) + two * tf.cast(off_term, dtype)


def logdet_from_precision(J: BlockTriDiagMatrix) -> tf.Tensor:
    """
    Minimal correct logdet via dense Cholesky:
      log|K| = -log|J|.
    """
    # TODO: If N is large, replace with structured block-tridiag Cholesky.
    J_dense = J.to_dense()
    J_dense = sym(J_dense) + default_jitter() * tf.eye(tf.shape(J_dense)[0], dtype=J_dense.dtype)
    L = tf.linalg.cholesky(J_dense)
    d = tf.linalg.diag_part(L)
    logdet_J = tf.cast(2.0, d.dtype) * tf.reduce_sum(tf.math.log(tf.maximum(d, default_jitter())))
    return -logdet_J


def quad_form_precision(J: BlockTriDiagMatrix, x: tf.Tensor) -> tf.Tensor:
    """
    Compute x^T J x using J.quad_form.
    """
    return J.quad_form(x)


def gauss_markov_kl(q_mean: tf.Tensor, q_cov: BlockTriDiagCovariance, J: BlockTriDiagMatrix) -> tf.Tensor:
    """
    KL[q || p] where:
      q(x) = N(q_mean, q_cov)
      p(x) = N(0, K), with J = K^{-1} block tri-diagonal.

    With new dataclasses:
      - q_cov is the covariance S directly (not a Cholesky factor)
      - J is the precision matrix directly (not a Cholesky factor)
    """
    q_mean = tf.convert_to_tensor(q_mean)

    # N,P as tensors; use tf.shape for graph mode
    N = tf.shape(q_mean)[0]

    # x^T J x
    mahalanobis = quad_form_precision(J, q_mean)

    # log determinants
    logdet_K = logdet_from_precision(J)   # log|K|
    logdet_S = q_cov.logdet

    # trace(J S)
    trace_term = calc_trace_J_S(J, q_cov)

    twoKL = - logdet_S + logdet_K - tf.cast(N, logdet_S.dtype) + trace_term + mahalanobis
    return 0.5 * twoKL
