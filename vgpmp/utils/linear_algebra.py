from dataclasses import dataclass
from typing import Optional, Union
import tensorflow as tf

from gpflow.config import default_jitter

from .math import softplus_inverse


def sym(A: tf.Tensor) -> tf.Tensor:
    return 0.5 * (A + tf.transpose(A))


# Helper to undo the softplus used in _ensure_lower_with_positive_diag so that
# the transformed diagonal matches a target Cholesky block.
def invert_softplus_diag(L_block: tf.Tensor) -> tf.Tensor:
    d = tf.linalg.diag_part(L_block)
    d_raw = softplus_inverse(tf.maximum(d, default_jitter()))
    return L_block - tf.linalg.diag(d) + tf.linalg.diag(d_raw)


def _ensure_lower_with_positive_diag(L: tf.Tensor) -> tf.Tensor:
    L = tf.linalg.band_part(L, -1, 0)
    d = tf.linalg.diag_part(L)
    d_pos = tf.nn.softplus(d) + default_jitter()
    return L - tf.linalg.diag(d) + tf.linalg.diag(d_pos)


@dataclass
class Covariance:
    """
    Block-symmetric covariance matrix K (SPD), stored in block form.

    Representation:
      - diags     : (N, P, P)  diagonal blocks K_{i,i}
      - off_diags : (N*(N-1)//2, P, P) lower-triangular blocks K_{i,j} for i>j

    Symmetry convention:
      - K_{j,i} = K_{i,j}^T
    """
    diags: tf.Tensor
    off_diags: tf.Tensor

    def __post_init__(self):
        tf.debugging.assert_rank(self.diags, 3, message="diags must be rank-3 (N,P,P)")
        tf.debugging.assert_rank(self.off_diags, 3, message="off_diags must be rank-3 (M,P,P)")
        tf.debugging.assert_equal(tf.shape(self.diags)[1], tf.shape(self.diags)[2], message="diag blocks must be square")
        tf.debugging.assert_equal(tf.shape(self.off_diags)[1], tf.shape(self.off_diags)[2], message="off blocks must be square")
        tf.debugging.assert_equal(tf.shape(self.diags)[1], tf.shape(self.off_diags)[1], message="block sizes must match")

    @staticmethod
    def _off_index(i: Union[int, tf.Tensor], j: Union[int, tf.Tensor]) -> tf.Tensor:
        """
        Map (i,j) with 0 <= j < i < N to row-major index in off_diags.
        """
        i = tf.convert_to_tensor(i)
        j = tf.convert_to_tensor(j)
        return i * (i - 1) // 2 + j

    @property
    def N(self) -> tf.Tensor:
        return tf.shape(self.diags)[0]

    @property
    def P(self) -> tf.Tensor:
        return tf.shape(self.diags)[1]

    def block(self, i: Union[int, tf.Tensor], j: Union[int, tf.Tensor]) -> tf.Tensor:
        """
        Return block K_{i,j} of shape (P,P), using symmetry.
        """
        i = tf.convert_to_tensor(i)
        j = tf.convert_to_tensor(j)

        def diag_case():
            return self.diags[i]

        def lower_case():
            idx = self._off_index(i, j)
            return self.off_diags[idx]

        def upper_case():
            idx = self._off_index(j, i)
            return tf.transpose(self.off_diags[idx])

        return tf.cond(
            tf.equal(i, j),
            diag_case,
            lambda: tf.cond(tf.greater(i, j), lower_case, upper_case),
        )

    def get_prev_next_block(self, i: Union[int, tf.Tensor]) -> tf.Tensor:
        """
        Return the 2x2 block submatrix:
            [ K_{i,i}     K_{i,i+1} ]
            [ K_{i+1,i}   K_{i+1,i+1} ]
        Shape: (2P, 2P)
        """
        i = tf.convert_to_tensor(i)
        N = self.N

        tf.debugging.assert_greater_equal(i, 0)
        tf.debugging.assert_less(i + 1, N)

        Kii = self.block(i, i)
        Kip1 = self.block(i, i + 1)
        Kp1i = tf.transpose(Kip1)
        Kp1p1 = self.block(i + 1, i + 1)

        top = tf.concat([Kii, Kip1], axis=1)
        bot = tf.concat([Kp1i, Kp1p1], axis=1)
        return tf.concat([top, bot], axis=0)

    def to_dense(self) -> tf.Tensor:
        """
        Materialize full dense matrix of shape (N*P, N*P).
        """
        # TODO: O((NP)^2) memory/time; avoid for large N.
        N = self.N
        P = self.P

        # Build block rows (Python-level loops are fine if N is static; if N is dynamic, prefer tf.while_loop).
        # This version is eager-friendly and works in tf.function if N is known at trace time.
        blocks = []
        for i in range(int(self.diags.shape[0])):  # requires static N
            row_blocks = []
            for j in range(int(self.diags.shape[0])):
                row_blocks.append(self.block(i, j))
            blocks.append(tf.concat(row_blocks, axis=1))
        return tf.concat(blocks, axis=0)

    @property
    def logdet(self) -> tf.Tensor:
        """
        log|K| using a numerically stable Cholesky on the dense matrix.
        """
        # TODO: If N is large, you should implement a structured logdet instead of densifying.
        K = self.to_dense()
        K = sym(K) + default_jitter() * tf.eye(tf.shape(K)[0], dtype=K.dtype)
        L = tf.linalg.cholesky(K)
        d = tf.linalg.diag_part(L)
        return tf.cast(2.0, d.dtype) * tf.reduce_sum(tf.math.log(tf.maximum(d, default_jitter())))

    @classmethod
    def build_from_anchored_covariance(cls, A: "AnchoredCovariance") -> "Covariance":
        """
        Reconstruct K from an AnchoredCovariance, by inserting the first/last diagonal blocks.
        """
        P = A.block_size
        anchor_vars = A.anchor_vars  # (P,)

        N = tf.shape(A.mid_diags)[0] + 2

        K00 = tf.linalg.diag(anchor_vars)  # (P,P)
        KNN = tf.linalg.diag(anchor_vars)  # (P,P)

        diags = tf.concat([K00[None, :, :], A.mid_diags, KNN[None, :, :]], axis=0)  # (N,P,P)
        return cls(diags=diags, off_diags=A.off_diags)


@dataclass
class AnchoredCovariance:
    """
    Covariance matrix with the *first and last* diagonal blocks removed.

    mid_diags: (N-2, P, P)  corresponds to K_{1,1}..K_{N-2,N-2}
    off_diags: (N*(N-1)//2, P, P) stores all lower-triangular off-diagonal blocks K_{i,j}, i>j
    anchor_vars: (P,) variances used to reconstruct K_{0,0} and K_{N-1,N-1} as diag(anchor_vars)
    """
    mid_diags: tf.Tensor
    off_diags: tf.Tensor
    anchor_vars: tf.Tensor

    flat_tensor: Optional[tf.Tensor] = None

    def __post_init__(self):
        tf.debugging.assert_rank(self.mid_diags, 3, message="mid_diags must be rank-3 (K, P, P)")
        tf.debugging.assert_rank(self.off_diags, 3, message="off_diags must be rank-3 (M, P, P)")
        tf.debugging.assert_equal(tf.shape(self.mid_diags)[1], tf.shape(self.mid_diags)[2],
                                  message="mid_diags blocks must be square (P, P)")
        tf.debugging.assert_equal(tf.shape(self.off_diags)[1], tf.shape(self.off_diags)[2],
                                  message="off_diags blocks must be square (P, P)")
        tf.debugging.assert_equal(tf.shape(self.mid_diags)[1], tf.shape(self.off_diags)[1],
                                  message="mid_diags and off_diags must share same block size P")
        tf.debugging.assert_rank(self.anchor_vars, 1, message="anchor_vars must be rank-1 (P,)")

        if self.flat_tensor is None:
            self.flat_tensor = self._build_flat_tensor()

    def _build_flat_tensor(self) -> tf.Tensor:
        diag_vec = tf.reshape(self.mid_diags, (-1,))
        off_vec = tf.reshape(self.off_diags, (-1,))
        return tf.concat([diag_vec, off_vec], axis=0)

    @property
    def block_size(self) -> tf.Tensor:
        return tf.shape(self.mid_diags)[1]

    @property
    def mid_count(self) -> tf.Tensor:
        return tf.shape(self.mid_diags)[0]

    @property
    def off_count(self) -> tf.Tensor:
        return tf.shape(self.off_diags)[0]

    @classmethod
    def build_from_flat_tensor(cls, N, P, flat_tensor: tf.Tensor, anchor_vars: tf.Tensor):
        diag_count = N - 2
        off_count = N * (N - 1) // 2
        block_elems = P * P

        diag_elems = diag_count * block_elems
        off_elems = off_count * block_elems

        diag_vec = flat_tensor[:diag_elems]
        off_vec = flat_tensor[diag_elems: diag_elems + off_elems]

        diag_stack = tf.reshape(diag_vec, (diag_count, P, P))
        off_stack = tf.reshape(off_vec, (off_count, P, P))

        return cls(mid_diags=diag_stack, off_diags=off_stack, anchor_vars=anchor_vars, flat_tensor=flat_tensor)

    @classmethod
    def build_from_unconstrained(cls, K: Covariance, anchor_vars: tf.Tensor):
        """
        Remove the first and last diagonal blocks:
          mid_diags = K.diags[1:-1]
          off_diags = K.off_diags (unchanged)
        """
        mid_diags = K.diags[1:-1]
        off_diags = K.off_diags
        return cls(mid_diags=mid_diags, off_diags=off_diags, anchor_vars=anchor_vars)


@dataclass
class BlockTriDiagMatrix:
    """
    Block tri-diagonal matrix A with symmetric off-diagonals:
      - diag blocks: A_{i,i}
      - sub blocks : A_{i+1,i}
      - sup blocks : A_{i,i+1} = (A_{i+1,i})^T

    diags    : (N,   P, P)
    sub_diags: (N-1, P, P)
    """
    diags: tf.Tensor
    sub_diags: tf.Tensor

    def __post_init__(self):
        tf.debugging.assert_rank(self.diags, 3, message="diags must be rank-3 (N,P,P)")
        tf.debugging.assert_rank(self.sub_diags, 3, message="sub_diags must be rank-3 (N-1,P,P)")
        tf.debugging.assert_equal(tf.shape(self.diags)[1], tf.shape(self.diags)[2], message="diag blocks must be square")
        tf.debugging.assert_equal(tf.shape(self.sub_diags)[1], tf.shape(self.sub_diags)[2], message="sub blocks must be square")
        tf.debugging.assert_equal(tf.shape(self.diags)[1], tf.shape(self.sub_diags)[1], message="block sizes must match")
        tf.debugging.assert_equal(tf.shape(self.sub_diags)[0] + 1, tf.shape(self.diags)[0],
                                  message="sub_diags must have length N-1")

    @property
    def num_diag_blocks(self) -> tf.Tensor:
        return tf.shape(self.diags)[0]

    @property
    def block_size(self) -> tf.Tensor:
        return tf.shape(self.diags)[1]

    @property
    def sup_diags(self) -> tf.Tensor:
        return tf.transpose(self.sub_diags, perm=[0, 2, 1])

    def matvec(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute y = A x for x shape (N,P).
        """
        x = tf.convert_to_tensor(x)
        tf.debugging.assert_rank(x, 2, message="x must be rank-2 (N,P)")
        tf.debugging.assert_equal(tf.shape(x)[0], self.num_diag_blocks, message="x has wrong N")
        tf.debugging.assert_equal(tf.shape(x)[1], self.block_size, message="x has wrong P")

        N = self.num_diag_blocks

        # diag part
        y = tf.linalg.matvec(self.diags, x)  # (N,P)

        # lower part contributes to rows 1..N-1: + A_{i,i-1} x_{i-1}
        y_lower = tf.linalg.matvec(self.sub_diags, x[:-1])  # (N-1,P)
        y = tf.concat([y[:1], y[1:] + y_lower], axis=0)

        # upper part contributes to rows 0..N-2: + A_{i,i+1} x_{i+1}
        y_upper = tf.linalg.matvec(self.sup_diags, x[1:])  # (N-1,P)
        y = tf.concat([y[:-1] + y_upper, y[-1:]], axis=0)

        return y

    def quad_form(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute x^T A x for x shape (N,P).
        """
        y = self.matvec(x)
        return tf.reduce_sum(x * y)

    def to_dense(self) -> tf.Tensor:
        """
        Materialize full dense (N*P, N*P).
        Requires static N at trace time (like the Covariance.to_dense above).
        """
        N = int(self.diags.shape[0])
        P = int(self.diags.shape[1])

        rows = []
        for i in range(N):
            row_blocks = []
            for j in range(N):
                if i == j:
                    Bij = self.diags[i]
                elif j == i - 1:
                    Bij = self.sub_diags[j]          # A_{i,i-1} = sub[j] where j=i-1
                elif j == i + 1:
                    Bij = tf.transpose(self.sub_diags[i])  # A_{i,i+1} = sub[i]^T
                else:
                    Bij = tf.zeros((P, P), dtype=self.diags.dtype)
                row_blocks.append(Bij)
            rows.append(tf.concat(row_blocks, axis=1))
        return tf.concat(rows, axis=0)


@dataclass
class BlockBiDiagFactor:
    """
    Block-bidiagonal factor L (lower), defining S = L L^T.

    L has:
      - diags:     (N,P,P) blocks L_{i,i} (lower-triangular with positive diag)
      - sub_diags: (N-1,P,P) blocks L_{i+1,i}

    This implies S is block-tridiagonal:
      S_{i,i}     = L_{i,i}L_{i,i}^T + (i>0) L_{i,i-1}L_{i,i-1}^T
      S_{i,i-1}   = L_{i,i-1}L_{i-1,i-1}^T
      S_{i,i+1}   = S_{i+1,i}^T
    """
    diags_raw: tf.Tensor
    sub_diags_raw: tf.Tensor

    def diags(self) -> tf.Tensor:
        return tf.map_fn(
            _ensure_lower_with_positive_diag,
            self.diags_raw, fn_output_signature=self.diags_raw.dtype
        )

    @property
    def sub_diags(self) -> tf.Tensor:
        return self.sub_diags_raw
    

@dataclass
class BlockTriDiagCovariance:
    """
    Block-tridiagonal covariance S.

      diags     : (N,P,P) blocks S_{i,i}
      sub_diags : (N-1,P,P) blocks S_{i+1,i}
    """
    diags: tf.Tensor
    sub_diags: tf.Tensor

    @property
    def N(self) -> tf.Tensor:
        return tf.shape(self.diags)[0]

    @property
    def P(self) -> tf.Tensor:
        return tf.shape(self.diags)[1]
    
    def block(self, i: tf.Tensor, j: tf.Tensor) -> tf.Tensor:
        """
        Block lookup.
        i, j: shape (B,) int32/int64
        returns: (B, P, P)

        Supports:
          - i == j       -> diags[i]
          - i == j + 1   -> sub_diags[j]       (S_{j+1,j})
          - j == i + 1   -> transpose(sub_diags[i]) (S_{i,i+1})
          - else         -> zeros
        """
        i = tf.convert_to_tensor(i)
        j = tf.convert_to_tensor(j)
        i = tf.reshape(i, [-1])
        j = tf.reshape(j, [-1])

        B = tf.shape(i)[0]
        P = tf.shape(self.diags)[1]
        dtype = self.diags.dtype

        # Masks
        diag_mask  = tf.equal(i, j)                 # i == j
        sub_mask   = tf.equal(i, j + 1)             # i == j+1
        super_mask = tf.equal(j, i + 1)             # j == i+1

        N = tf.shape(self.diags)[0]
        i_safe = tf.clip_by_value(i, 0, N - 1)
        diag_blocks = tf.gather(self.diags, i_safe)  # (B,P,P)

        Nm1 = tf.shape(self.sub_diags)[0]            # N-1
        j_safe = tf.clip_by_value(j, 0, Nm1 - 1)
        sub_blocks = tf.gather(self.sub_diags, j_safe)  # (B,P,P)

        i_safe2 = tf.clip_by_value(i, 0, Nm1 - 1)
        super_blocks = tf.transpose(tf.gather(self.sub_diags, i_safe2), perm=[0, 2, 1])  # (B,P,P)

        zeros = tf.zeros([B, P, P], dtype=dtype)

        out = tf.where(diag_mask[:, None, None], diag_blocks, zeros)
        out = tf.where(sub_mask[:, None, None], sub_blocks, out)
        out = tf.where(super_mask[:, None, None], super_blocks, out)
        return out

    def diag_block(self, i: tf.Tensor) -> tf.Tensor:
        """
        Convenience: returns S_{i,i} for i: (B,) -> (B,P,P)
        """
        i = tf.reshape(tf.convert_to_tensor(i), [-1])
        N = tf.shape(self.diags)[0]
        i = tf.clip_by_value(i, 0, N - 1)
        return tf.gather(self.diags, i)

    def get_prev_next_block(self, i: tf.Tensor) -> tf.Tensor:
        """
        i: (B,) with values in [0..N-2]
        returns: (B, 2P, 2P) where each is:
          [[S_{i,i},   S_{i,i+1}],
           [S_{i+1,i}, S_{i+1,i+1}]]
        """
        i = tf.reshape(tf.convert_to_tensor(i), [-1])
        B = tf.shape(i)[0]
        P = tf.shape(self.diags)[1]
        N = tf.shape(self.diags)[0]

        # Ensure i in [0, N-2]
        i = tf.clip_by_value(i, 0, N - 2)

        Sii = tf.gather(self.diags, i)         # (B,P,P)
        Sip1ip1 = tf.gather(self.diags, i + 1) # (B,P,P)

        # S_{i+1,i} stored in sub_diags[i]
        Sip1_i = tf.gather(self.sub_diags, i)  # (B,P,P)

        # S_{i,i+1} = transpose(S_{i+1,i})
        Si_ip1 = tf.transpose(Sip1_i, perm=[0, 2, 1])  # (B,P,P)

        top = tf.concat([Sii, Si_ip1], axis=2)      # (B,P,2P)
        bot = tf.concat([Sip1_i, Sip1ip1], axis=2)  # (B,P,2P)
        S2 = tf.concat([top, bot], axis=1)          # (B,2P,2P)
        return S2
    
    @property
    def logdet(self) -> tf.Tensor:
        """
        log|S| for block-tridiagonal SPD S with:
        diags    : (N,P,P) blocks S_{i,i}
        sub_diags: (N-1,P,P) blocks S_{i+1,i}

        Complexity: O(N * P^3)
        """
        diags = tf.convert_to_tensor(self.diags)
        sub_diags = tf.convert_to_tensor(self.sub_diags)

        N = tf.shape(diags)[0]
        P = tf.shape(diags)[1]
        dtype = diags.dtype

        I = tf.eye(P, dtype=dtype)
        jitter = tf.cast(default_jitter(), dtype) * I

        # i = 0
        S00 = sym(diags[0]) + jitter
        L_prev = tf.linalg.cholesky(S00)
        d0 = tf.linalg.diag_part(L_prev)
        acc0 = tf.cast(2.0, dtype) * tf.reduce_sum(tf.math.log(tf.maximum(d0, tf.cast(default_jitter(), dtype))))

        def cond(i, L_prev, acc):
            return i < N

        def body(i, L_prev, acc):
            # E = S_{i, i-1}
            E = sub_diags[i - 1]  # (P,P)

            # L_{i,i-1} = E * inv(L_{i-1,i-1}^T)
            X_T = tf.linalg.triangular_solve(L_prev, tf.transpose(E), lower=True)  # (P,P)
            L_i_im1 = tf.transpose(X_T)

            # Schur complement for diagonal block i
            Si = sym(diags[i] - tf.matmul(L_i_im1, L_i_im1, transpose_b=True)) + jitter
            L_i = tf.linalg.cholesky(Si)

            di = tf.linalg.diag_part(L_i)
            acc = acc + tf.cast(2.0, dtype) * tf.reduce_sum(tf.math.log(tf.maximum(di, tf.cast(default_jitter(), dtype))))

            return i + 1, L_i, acc

        # If N==1, loop body won't run, and we return acc0.
        _, _, acc = tf.while_loop(cond, body, loop_vars=[tf.constant(1, tf.int32), L_prev, acc0])
        return acc
    

def covariance_from_bidiag_factor(L: BlockBiDiagFactor) -> BlockTriDiagCovariance:
    Ld = L.diags()            # (N,P,P)
    Ls = L.sub_diags          # (N-1,P,P)

    # Diagonal blocks:
    # S_0 = L00 L00^T
    # S_i = Lii Lii^T + L_{i,i-1} L_{i,i-1}^T
    S0 = tf.matmul(Ld[0], Ld[0], transpose_b=True)  # (P,P)

    S_mid = tf.matmul(Ld[1:], Ld[1:], transpose_b=True) + tf.matmul(Ls, Ls, transpose_b=True)  # (N-1,P,P)
    S_diags = tf.concat([S0[None, :, :], S_mid], axis=0)

    # Sub-diagonal blocks:
    # S_{i,i-1} = L_{i,i-1} L_{i-1,i-1}^T
    S_sub = tf.matmul(Ls, Ld[:-1], transpose_b=True)  # (N-1,P,P)

    # Symmetrize diagonal blocks for numerical consistency
    S_diags = 0.5 * (S_diags + tf.transpose(S_diags, perm=[0, 2, 1]))

    return BlockTriDiagCovariance(diags=S_diags, sub_diags=S_sub)
