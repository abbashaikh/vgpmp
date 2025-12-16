import tensorflow as tf
from gpflow.config import default_float, default_jitter

from .utils.linear_algebra import Cholesky, BlockTriDiagCholesky


def _off_index(i: int, j: int) -> int:
      """
      Map (i,j) with 0 <= j < i < N to the row-major index in the off-diagonal list.
      Blocks are appended for i=1..N-1, and within each i, j=0..i-1.

      O(N^2 P^3)
      """
      return i*(i-1)//2 + j


#TODO: check trace calculation equations
def calc_trace_Kinv_S(Kinv: BlockTriDiagCholesky, S: Cholesky):
      """
      K^{-1} = L L^T
      S = C C^T
      """
      N = Kinv.num_diag_blocks
      P = Kinv.block_size

      ## to access C blocks (row r, col c, with r>=c)
      def C_block(r: int, c: int) -> tf.Tensor:
            if r == c:
                  return S.diags[r]
            else:
                  return S.off_diags[_off_index(r, c)]
            
      ## \Sigma_r = sum_{i=0}^r C_{r,i} C_{r,i}^T
      Sigma = []
      for r in range(N):
            acc = tf.zeros((P, P), dtype=default_float())
            for i in range(r + 1):
                  B = C_block(r, i)
                  acc += tf.matmul(B, B, transpose_b=True)
            # for numerical consistency
            acc = 0.5 * (acc + tf.transpose(acc))
            Sigma.append(acc)

      ## \Xi_j = sum_{i=0}^j C_{j+1,i} C_{j,i}^T
      Xi = []
      for j in range(N - 1):
            acc = tf.zeros((P, P), dtype=default_float())
            for i in range(j + 1):
                  B_up = C_block(j + 1, i)
                  B_lo = C_block(j, i)
                  acc += tf.matmul(B_up, B_lo, transpose_b=True)
            Xi.append(acc)

      ## accumulate trace terms
      total = tf.zeros((), dtype=default_float())

      # trace(G_j \Sigma_j)
      for j in range(N):
        Dj = Kinv.diags[j]
        Gj = tf.matmul(Dj, Dj, transpose_b=True)      # (P,P)
        total += tf.reduce_sum(Gj * Sigma[j])

      # trace(H_j \Sigma_{j+1}) + 2*trace(T_j \Xi_j)
      for j in range(N - 1):
        Lj = Kinv.sub_diags[j]      # subdiag block (j+1, j)
        Dj = Kinv.diags[j]

        Hj = tf.matmul(Lj, Lj, transpose_b=True)      # (P,P)
        Tj = tf.matmul(Dj, Lj, transpose_b=True)      # (P,P)

        # tr(H_j \Sigma_{j+1})
        total += tf.reduce_sum(Hj * Sigma[j + 1])
        # 2*tr(T_j \Xi_j)
        total += tf.cast(2.0, default_float) * tf.reduce_sum(Tj * Xi[j])

      return total


def gauss_markov_kl(q_mean: tf.Tensor, q_cov: Cholesky, Kinv: BlockTriDiagCholesky):
    """
    Compute the KL divergence KL[q || p] between::

          q(x) = N(q_mu, q_sqrt^2)

    and::

          p(x) = N(0, K)
          where K is tridiagonal (Gauss-Markov)

    N : number of data points
    P : number of latent GPs

    Input:
        - q_mean : shape (N, P)
        - q_cov  : shape (block_anchored_cholesky_size(N, P),)
    """
    N, P = tf.shape(q_mean)

    # mahalanobis
    mahalanobis = Kinv.mahalanobis(q_mean)

    # log determinants
    logdet_K = - Kinv.logdet
    logdet_S = q_cov.logdet

    # trace(K^{-1} S)
    trace_term = calc_trace_Kinv_S(Kinv, q_cov)

    twoKL = logdet_S + logdet_K - N + trace_term + mahalanobis

    return 0.5 * twoKL