from dataclasses import dataclass
from typing import List, Optional, Tuple
import tensorflow as tf

from gpflow.config import default_float, default_jitter


@dataclass
class Cholesky:
    """
    diags       : List of N `P` by `P` tesnors
    off_diags   : List of N * (N-1) // 2 `P` by `P` tensors
    """
    diags: List[tf.Tensor]
    off_diags: List[tf.Tensor]

    def _ensure_lower_with_positive_diag(self, diag_block: tf.Tensor) -> tf.Tensor:
        """Force lower-triangular and strictly positive diagonal via softplus+jitter"""
        diag_block = tf.linalg.band_part(diag_block, -1, 0)
        d = tf.linalg.diag_part(diag_block)
        d_pos = tf.nn.softplus(d) + default_jitter()
        return diag_block - tf.linalg.diag(d) + tf.linalg.diag(d_pos)
    
    def _off_index(self, i: int, j: int) -> int:
        """
        Map (i,j) with 0 <= j < i < N to the row-major index in the off-diagonal list.
        Blocks are appended for i=1..N-1, and within each i, j=0..i-1.
        """
        return i*(i-1)//2 + j
    
    @classmethod
    def build_from_anchored_cholesky(cls, A: "AnchoredCholesky") -> "Cholesky":
        N = len(A.mid_diags) + 2
        P = A.block_size
        anchor_vars = A.anchor_vars

        # diag blocks list
        diags: List[tf.Tensor] = [None]*N
        # L_00 anchored
        diags[0] = tf.linalg.diag(tf.sqrt(anchor_vars))
        # internal diag blocks L_ii
        for i in range(1, N-1):
            diags[i] = cls._ensure_lower_with_positive_diag(A.mid_diags[i-1])
        # L_NN anchored
        acc = 0
        for j in range(0, N-1):
            Lij = A.off_diags[cls._off_index(N-1, j)]
            # diag(B B^T) as row-wise sum of squares
            d = tf.reduce_sum(tf.square(Lij), axis=1)                       # shape (P,)
            acc = acc + d
        # initialize L_NN
        #TODO: keep an eye on this diff!
        diff = tf.maximum(anchor_vars - acc, default_jitter())
        L_NN_init = tf.linalg.diag(tf.sqrt(diff))
        diags[N-1] = cls._ensure_lower_with_positive_diag(L_NN_init)
        # scale last row of L to provide anchored goal covariance
        last_row_sum = acc + tf.reduce_sum(tf.square(diags[N-1]), axis=1)

        scale = tf.sqrt(anchor_vars / (last_row_sum + default_jitter()))    # shape (P,)
        D = tf.linalg.diag(scale)

        scaled_off_blocks = list(A.off_diags)
        for j in range(0, N-1):
            idx = cls._off_index(N-1, j)
            scaled_off_blocks.off_diags[idx] = D @ A.off_diags[idx]
        diags[N-1] = D @ diags[N-1]
        diags[N-1] = cls._ensure_lower_with_positive_diag(diags[N-1])

        return cls(diags=diags, off_diags=scaled_off_blocks)
    
    @property
    def logdet(self) -> tf.Tensor:
        """
        Return log|A|,
        for block tri-diagonal A.
        log|A| = 2 * (sum_i log|L_ii|), where A = L*L^T.
        """
        diag = tf.stack(self.diags, axis=0)
        d = tf.linalg.diag_part(diag)
        d_safe = tf.maximum(d, default_jitter())

        return tf.cast(2.0, dtype=default_float()) * tf.reduce_sum(tf.math.log(d_safe))
    
    def get_row(self, k: int) -> List[tf.Tensor]:
        """
        Return the k-th block row of the Cholesky factor as a list (excluding zero blocks)
        [L_{k,0}, ..., L_{k,k}]
        """
        N = len(self.diags)

        # off diag blocks in row k: L_{k1}, ..., L_{k,k-1}
        blocks = [ self.off_diags[self._off_index(k, j)] for j in range(k) ]

        # Append diagonal block L_{kk}
        blocks.append(self.diags[k])

        return blocks
    
    def get_prev_next_block(self, i: int) -> tf.Tensor:
        """
        Construct the 2×2 symmetric block matrix:
            [ S_ii        S_i_ip1     ]
            [ S_i_ip1^T   S_ip1_ip1   ]

        Each block is P×P, output is (2P)×(2P).
        """
        N = len(self.diags)
        if i < 0 or (i + 1) >= N:
            raise IndexError(f"i={i} out of range for N={N} (need 0 <= i < N-1).")
        
        row_i = self.get_row(i)
        row_ip1 = self.get_row(i + 1)

        # S_{ii} = sum_{k=0..i} L_{i,k} L_{i,k}^T
        S_ii = tf.add_n([tf.matmul(B, B, transpose_b=True) for B in row_i])

        # S_{i,i+1} = sum_{k=0..i} L_{i,k} L_{i+1,k}^T
        S_i_ip1 = tf.add_n([tf.matmul(row_i[k], row_ip1[k], transpose_b=True) for k in range(i + 1)])
        # S_{i+1,i+1} = sum_{k=0..i+1} L_{i+1,k} L_{i+1,k}^T
        S_ip1_ip1 = tf.add_n([tf.matmul(B, B, transpose_b=True) for B in row_ip1])

        output = tf.concat([
            tf.concat([S_ii, S_i_ip1], axis=1),
            tf.concat([tf.transpose(S_i_ip1), S_ip1_ip1], axis=1)
        ], axis=0)
        return output

@dataclass
class AnchoredCholesky:
    """
    mid_diags: List of (N-2) `P` by `P` tesnors
    off_diags: List of N * (N-1) // 2 `P` by `P` tensors
    """
    mid_diags: List[tf.Tensor]
    off_diags: List[tf.Tensor]
    anchor_vars: tf.Tensor

    flat_tensor: Optional[tf.Tensor] = None

    def __post_init__(self):
        if self.flat_tensor is None:
            self.flat_tensor = self._build_flat_tensor()

    def _build_flat_tensor(self) -> tf.Tensor:
        '''return: 1-D tensor of length [(N-2) + N*(N-1)//2] * P * P'''
        P = self.block_size

        def _flatten_list(blocks: List[tf.Tensor]) -> tf.Tensor:
            # Flatten each to (P*P,) and concatenate
            flat_list = [tf.reshape(b, (-1,)) for b in blocks]
            return tf.concat(flat_list, axis=0)
        
        diag_vec = _flatten_list(self.mid_diags)
        off_vec  = _flatten_list(self.off_diags)
        return tf.concat([diag_vec, off_vec], axis=0)
    
    @property
    def block_size(self):
        return tf.shape(self.mid_diags[0])[0]
    
    @classmethod
    def build_from_flat_tensor(cls, N, P, flat_tensor: tf.Tensor, anchor_vars:tf.Tensor):
        diag_count = N - 2
        off_count  = N * (N - 1) // 2
        block_elems = P * P

        diag_elems = diag_count * block_elems
        off_elems = off_count * block_elems

        diag_vec = flat_tensor[:diag_elems]
        off_vec  = flat_tensor[diag_elems: diag_elems + off_elems]

        diag_stack = tf.reshape(diag_vec, (diag_count, P, P))
        mid_diags = [diag_stack[i] for i in range(diag_count)]

        off_stack = tf.reshape(off_vec, (off_count, P, P))
        off_diags = [off_stack[i] for i in range(off_count)]

        return cls(mid_diags=mid_diags, off_diags=off_diags, anchor_vars=anchor_vars, flat_tensor=flat_tensor)
    
    @classmethod
    def build_from_unconstrained(cls, A: "Cholesky", anchor_vars: tf.Tensor):
        N = len(A.diags)
        P = tf.shape(A.diags[0])[0]

        mid_diags = A.diags[1:N - 1]
        off_diags = A.off_diags

        return cls(mid_diags=mid_diags, off_diags=off_diags, anchor_vars=anchor_vars)


@dataclass
class BlockTriDiagCholesky:
    """
    Stores and operates on cholesky of a block-tridiagonal matrix
    with N diagonal blocks of size P×P and (N-1) lower off-diagonal blocks (i+1,i).
    ---------------
    diags[i]     : shape (P,P)
    sub_diags[i] : shape (P,P)
    ---------------
    num_blocks : int    # N
    block_size : int    # P
    """
    diags: List[tf.Tensor]
    sub_diags: List[tf.Tensor]
    logdet: Optional[tf.Tensor] = None

    def __post_init__(self):
        if self.logdet is None:
            self.logdet = self._calc_logdet()
    
    @property
    def num_diag_blocks(self) -> int:
        return len(self.diags)

    @property
    def block_size(self) -> int:
        return int(self.diags[0].shape[-1])
    
    # @tf.function
    def mahalanobis(self, x: tf.Tensor) -> tf.Tensor:
        """
        Return "x^T A x" for block tri-diagonal A

        Input
            x: shape (N, P)
        """
        N, P = self.num_diag_blocks, self.block_size
        
        diag = tf.stack(self.diags, axis=0)
        off = tf.stack(self.sub_diags, axis=0)

        # L_ii^T x_i
        Lx_diag = tf.linalg.matvec(tf.transpose(diag, perm=[0, 2, 1]), x)           # shape (N, P)
        # L_{i,i-1}^T x_{i-1}
        Lx_off_diag = tf.linalg.matvec(tf.transpose(off, perm=[0, 2, 1]), x[:-1])   # shape (N-1, P)
    
        y = tf.tensor_scatter_nd_add(
            Lx_diag,
            indices=tf.reshape(tf.range(1, N), (-1, 1)),
            updates=Lx_off_diag
        )

        # x^T A x = ||y||^2
        return tf.reduce_sum(tf.square(y))
    
    # TODO: tf.function needed? If yes will need an implementation not inheriting self
    # @tf.function
    def _calc_logdet(self) -> tf.Tensor:
        """
        Return log|A|,
        for block tri-diagonal A.
        log|A| = 2 * (sum_i log|L_ii|), where A = L*L^T.
        """
        diag = tf.stack(self.diags, axis=0)
        d = tf.linalg.diag_part(diag)
        d_safe = tf.maximum(d, default_jitter())

        return tf.cast(2.0, dtype=default_float()) * tf.reduce_sum(tf.math.log(d_safe))

    def thomas_solve(self, b: tf.Tensor) -> tf.Tensor:
        """
        Given the equation: Ax = b
        solve for "x" for block tri-diagonal A
        """
        N, D = self.num_blocks - 1, self.block_size
        # Forward sweep
        L = []  # subdiagonal factors
        A_tilde = [self.diag[0]]
        b_tilde = [b[0]]
        for i in range(N):
            Li = tf.linalg.solve(A_tilde[i], tf.transpose(self.off[i]))
            L.append(Li)
            Ai_tilde = self.diag[i+1] - tf.matmul(self.off[i], Li)
            A_tilde.append(Ai_tilde)
            bi_tilde = b[i + 1] - tf.linalg.matvec(self.off[i], tf.linalg.solve(A_tilde[i], b_tilde[i]))
            b_tilde.append(bi_tilde)

        # Back substitution
        x = [None] * (N + 1)
        x[N] = tf.linalg.solve(A_tilde[N], b_tilde[N])
        for i in reversed(range(N)):
            rhs = b_tilde[i] - tf.linalg.matvec(tf.transpose(self.off[i]), x[i + 1])
            x[i] = tf.linalg.solve(A_tilde[i], rhs)
        x = tf.stack(x, axis=0)

        return tf.reshape(x, [(N + 1) * D])

    def chol_solve(self, b: tf.Tensor) -> tf.Tensor:
        """
        Solve Ax = b for block tridiagonal A
        using L, where A = L*L^T (cholesky decomposition)
        """
        N = self.num_blocks - 1
        L = self.chol_block_tridiag()


        # Forward solve: L*y = b
        y = [None]*(N+1)
        y[0] = tf.linalg.triangular_solve(L.diag[0], b[0], lower=True)
        for i in range(1, N+1):
            rhs = b[i] - tf.matmul(L.off[i-1], y[i-1])
            y[i] = tf.linalg.triangular_solve(L.diag[i], rhs, lower=True)

        # Back solve: L^T*x = y
        x = [None]*(N+1)
        x[N] = tf.linalg.triangular_solve(tf.transpose(L.diag[N]), y[N], lower=False)
        for i in range(N-1, -1, -1):
            rhs = y[i] - tf.matmul(tf.transpose(L.off[i]), x[i+1])
            x[i] = tf.linalg.triangular_solve(tf.transpose(L.diag[i]), rhs, lower=False)

        return tf.stack(x, axis=0)
    

def sym(A: tf.Tensor): 
    return 0.5 * (A + tf.transpose(A))