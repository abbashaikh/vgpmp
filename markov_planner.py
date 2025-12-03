import numpy as np
import tensorflow as tf
import gpflow as gf

from vgpmp.dynamics.constant_velocity import ConstantVelocityModel
from vgpmp.prior.markov_prior import GaussMarkovPrior
from vgpmp.old.likelihood import CollisionLikelihood
from vgpmp.models.markov_vgp import MarkovVGP

DTYPE = tf.float32

"""
Notation:
    D -> total number of states
    N -> number of time segments
"""

# TODO: implement natural gradients
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
    # system's degree of freedom
    dof = 2

    '''Train Data'''
    # start and goal positions
    q_start = np.array([0.0, 0.0], dtype=np.float64)
    q_goal  = np.array([10.0, 10.0], dtype=np.float64)
    # start and end time
    t0, tN  = 0.0, 1.0
    # number of interior support points (K)
    num_steps = 10
    ## Interior points
    # Times excluding start and end
    t_interior = np.linspace(t0, tN, num_steps + 1)[1:-1]
    X_in = t_interior.reshape(-1, 1).astype(np.float64)         # shape (K,1)
    # Dummy Y_in if your likelihood ignores Y
    Y_in = np.zeros((num_steps - 1, dof), dtype=np.float64)      # shape (K, dof)
    ## End points
    X_ep = np.array([[t0], [tN]], dtype=np.float64)             # shape (2,1)
    Y_ep = np.stack([q_start, q_goal], axis=0)                  # shape (2,dof)

    '''Test Data'''
    X_test = np.linspace(t0, tN, 20, dtype=np.float64).reshape(-1,1)

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

    model = MarkovVGP(
        data,
        kernel,
        prior,
        likelihood,
        num_samples=num_steps,
        num_latent_gps=dof
    )

    #TODO: set trainable parameters and assign appropriate priors to them

    '''Training loop'''
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

    '''Output'''
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




