import dataclasses
from typing import List
import tensorflow as tf

DTYPE = tf.float32


@dataclasses.dataclass
class BlockTriDiag:
    """
    Symmetric block-tridiagonal matrix with (N+1) diagonal blocks of size D×D
    and N lower off-diagonal blocks (i+1,i). Upper off-diagonals are implied
    by symmetry.
    ---------------
    diag[i]  : (D,D)
    off[i]   : (D,D)
    ---------------
    num_blocks : int    # N
    block_size : int    # D
    """
    diag: List[tf.Tensor]
    off: List[tf.Tensor]
    jitter: float = 0.0

    @property
    def num_blocks(self) -> int:
        return len(self.diag)

    @property
    def block_size(self) -> int:
        return int(self.diag[0].shape[-1])
    
    @classmethod
    def from_blocks(cls, diag: List[tf.Tensor], off: List[tf.Tensor]) -> "BlockTriDiag":
        return cls(diag=diag, off=off)

    def __post_init__(self):
        self._chol_ready = tf.Variable(False, trainable=False)
        self._chol_diag_cache = [tf.Variable(tf.eye(self.block_size, dtype=DTYPE), trainable=False) for _ in self.diag]
        self._chol_off_cache  = [tf.Variable(tf.zeros_like(offi), trainable=False) for offi in self.off]
        self._logdet_cache = tf.Variable(0.0, dtype=DTYPE, trainable=False)

    def build_cache(self):
        L = self._chol_no_cache()
        for v, src in zip(self._chol_diag_cache, L.diag): v.assign(src)
        for v, src in zip(self._chol_off_cache, L.off): v.assign(src)
        logdet = 2.0 * tf.add_n([tf.reduce_sum(tf.math.log(tf.linalg.diag_part(d))) for d in L.diag])
        self._logdet_cache.assign(logdet)
        self._chol_ready.assign(True)

    def quad_form(self, x: tf.Tensor) -> tf.Tensor:
        """
        Return "x^T*A*x" for block tri-diagonal A
        """
        N, D = self.num_blocks - 1, self.block_size
        x = tf.reshape(x, [N + 1, D])

        acc = tf.einsum('i,ij,j->', x[0], self.diag[0], x[0])
        for i in range(N):
            acc += tf.einsum('i,ij,j->', x[i+1], self.diag[i + 1], x[i+1])
            acc += 2.0 * tf.einsum('i,ij,j->', x[i], self.off[i], x[i + 1])
        return acc

    def trace_tridiag_mul(self, B: "BlockTriDiag") -> tf.Tensor:
        """
        Return tr(AB)
        For block-tridiagonal A,B:
            tr(A B) = sum_i tr(A_ii B_ii) + 2 * sum_i tr(A_{i,i+1} B_{i+1,i})
        """
        N = self.num_blocks - 1
        total = tf.constant(0.0, dtype=DTYPE)
        for i in range(N + 1):
            total += tf.linalg.trace(tf.matmul(self.diag[i], B.diag[i]))
        for i in range(N):
            total += 2.0 * tf.linalg.trace(tf.matmul(tf.transpose(self.off[i]), B.off[i]))
        return total

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
    
    def _chol(self):
        if self._chol_ready:
            return BlockTriDiag(
                diag=[tf.identity(v) for v in self._chol_diag_cache],
                off=[tf.identity(v) for v in self._chol_off_cache]
            )
        return self._chol_no_cache()
    
    def _chol_no_cache(self) -> "BlockTriDiag":
        """
        Block Cholesky for symmetric positive definite block-tridiagonal self.
        Produces L such that A = L L^T with:
            L_{ii}      = lower-triangular (Cholesky of Schur complement at step i)
            L_{i+1,i}   = A_{i+1,i} * L_{i,i}^{-T}
        """
        N = self.num_blocks - 1

        jitter = self.jitter
        I = tf.eye(self.block_size, dtype=self.diag[0].dtype)
        
        L_diag: List[tf.Tensor] = []
        L_off:  List[tf.Tensor] = []

        # i=0
        L00 = tf.linalg.cholesky(
            0.5 * (self.diag[0] + tf.transpose(self.diag[0])) + jitter * I
        )
        L_diag.append(L00)

        for i in range(N):
            Li = tf.linalg.triangular_solve(tf.transpose(L_diag[i]), self.off[i], lower=False)
            L_off.append(Li)

            # Schur complement for next diagonal
            S = 0.5 * (self.diag[i + 1] + tf.transpose(self.diag[i + 1])) - tf.matmul(Li, tf.transpose(Li))
            Lii = tf.linalg.cholesky(S + jitter * I)
            L_diag.append(Lii)

        L = BlockTriDiag(diag=L_diag, off=L_off)

        return L

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

    def logdet(self, cache_only=False) -> tf.Tensor:
        """
        Return log|A|,
        for block tri-diagonal A.
        log|A| = 2 * (sum_i log|L_ii|), where A = L*L^T.
        """
        if self._chol_ready:
            return tf.identity(self._logdet_cache)
        if cache_only:
            return self._logdet_cache  # used only after build_cache

        L = self._chol_block_tridiag_no_cache()
        return 2.0 * tf.add_n([tf.reduce_sum(tf.math.log(tf.linalg.diag_part(d))) for d in L.diag])