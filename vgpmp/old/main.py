import numpy as np
import tensorflow as tf
import gpflow as gf

from .dynamics import ConstantVelocityModel
from .prior import GaussMarkovPrior
from .likelihood import CollisionLikelihood
from .models import VGP

DTYPE = tf.float32

"""
Notation:
    D -> total number of states
    N -> number of time segments
"""

@tf.function(jit_compile=False)
def step(model, nsamples=32):
    with tf.GradientTape() as tape:
        loss = -model.elbo(nsamples=nsamples)
    grads = tape.gradient(loss, model.trainable_variables)
    # optional gradient clipping
    grads = [tf.clip_by_norm(g, 5.0) if g is not None else None for g in grads]
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# def sample_trajectories(model, nsamples=20):
#     # Draw xi ~ N(0, I), x = m + L_q xi
#     Lq = banded_to_dense(model.Lq_banded, N+1, D)
#     xi = tf.random.normal([(N+1)*D, nsamples], dtype=Lq.dtype)
#     x = tf.reshape(model.m, [(N+1)*D, 1]) + tf.matmul(Lq, xi)
#     x = tf.transpose(x)  # [S, (N+1)D]
#     return tf.reshape(x, [nsamples, N+1, D]).numpy()


if __name__=="__main__":
    # shape(times) = num of segments + 1
    times = tf.constant(np.linspace(0.0, 1.0, 51), dtype=DTYPE)

    dof = 2
    dynamics_model = ConstantVelocityModel(dof, q_acc=1e-12)

    # initial
    K0  = tf.linalg.diag(
        tf.concat([
            1e-3*tf.ones([dof], DTYPE),     # q0 variance
            1e-3*tf.ones([dof], DTYPE),     # v0 variance
        ], axis=0)
    )                                       # shape (D, D)
    mu0 = tf.zeros([2*dof], DTYPE)          # shape (D,)
    start = (K0, mu0) 

    q_goal = tf.concat([10.0, 10.0], axis=0)
    
    # terminal
    KN  = tf.linalg.diag(
        tf.concat([
            1e-3*tf.ones([dof], DTYPE),    # q0 variance
            1e-3*tf.ones([dof], DTYPE),    # v0 variance
        ], axis=0)
    )
    muN = tf.concat([
        q_goal,
        tf.zeros([dof], DTYPE)
    ], axis=0)
    end = (KN, muN)

    prior = GaussMarkovPrior(
        times,
        dynamics=dynamics_model,
        start=start,
        end=end
    )

    # Build likelihood for a single circular obstacle centered at (5, 5) with radius 2.0
    likelihood = CollisionLikelihood(
        N=prior.num_segments,
        D=prior.block_dim,
        obstacle_center=(5.0, 5.0),
        obstacle_radius=2.0,
        grid_size=10.0,
        epsilon=0.3,
        sigma_obs=0.05,
        sigma_box=0.05,
        lambda_collision=1.0,
        lambda_box=1.0,
        enforce_box=True,
    )

    model = VGP(prior, likelihood)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    for it in range(1000):
        loss_val = step(model, nsamples=32)
        if it % 100 == 0:
            tf.print("iter", it, "negELBO", loss_val)

    scipy_opt = gf.optimizers.Scipy()

    scipy_opt.minimize(
        model.objective_closure(nsamples=128),
        variables=model.trainable_variables,
        options=dict(maxiter=200)
    )

    ## Mean plan on the support grid
    plan = model.m.numpy()  # shape: (N+1, D)

    q_plan = plan[:, :dof]  # positions only
    v_plan = plan[:, dof:]  # velocities

    ## Covariances
    # N = plan.shape[0] - 1
    # D = plan.shape[1]
    # Lq_dense = banded_to_dense(model.Lq_banded, N+1, D)     # ((N+1)D, (N+1)D)
    # S_dense = tf.matmul(Lq_dense, Lq_dense, transpose_b=True)

    # # Extract diagonal blocks (D_state × D_state) → marginal covariances per time
    # marg_covs = []
    # for i in range(N+1):
    #     sl = slice(i*D, (i+1)*D)
    #     Si = S_dense[sl, sl]
    #     marg_covs.append(Si.numpy())

    # pos_std = []
    # for Si in marg_covs:
    #     pos_std.append(np.sqrt(np.diag(Si)[:dof]))  # std for q components
    # pos_std = np.stack(pos_std, axis=0)             # (N+1, d)

    # samples = sample_trajectories(model, nsamples=50)




