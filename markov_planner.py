from typing import Tuple
from tqdm import tqdm
import time

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_float

from vgpmp.dynamics.constant_velocity import ConstantVelocityModel
from vgpmp.kernels import GaussMarkovKernel
from vgpmp.likelihood import PlanningLikelihood
from vgpmp.posterior import GaussMarkovPosterior
from vgpmp.models.markov_vgp import MarkovVGP
from vgpmp.utils.plotting import plot_mean_and_obstacle

"""
Notation:
    D -> total number of states
    N -> number of time segments
"""

gpflow.config.set_default_float(tf.float64)
gpflow.config.set_default_jitter(1e-6)


def optimization_step(model, closure, optimizer, nsamples: int = 8):
    # Eager execution avoids tf.Variable creation issues for optimizer slots inside tf.function.
    with tf.GradientTape() as tape:
        loss = closure()
        # sum_var_exp, KL, elbo = model.elbo_terms(nsamples=nsamples)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss#, sum_var_exp, KL, elbo


def build_obstacle_trajectory(total_steps: int, center0: Tuple[float, float], velocity: Tuple[float, float], radius: float):
    """
    Construct a simple linear obstacle trajectory sampled at discrete steps.
    Returns centers with shape (total_steps, 1, dof) and radii with shape (total_steps, 1).
    """
    centers = np.zeros((total_steps, 1, 2), dtype=np.float64)
    radii = np.full((total_steps, 1), radius, dtype=np.float64)

    base = np.asarray(center0, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)

    for k in range(total_steps):
        centers[k, 0, :] = base + vel * float(k)

    return centers, radii


def slice_window(arr: np.ndarray, start_idx: int, window_len: int) -> np.ndarray:
    """
    Take a window from a time-indexed array; repeat the last value if we fall off the end.
    """
    window = arr[start_idx:start_idx + window_len]
    if window.shape[0] == 0:
        window = np.repeat(arr[-1:, ...], window_len, axis=0)
    elif window.shape[0] < window_len:
        pad = np.repeat(window[-1:, ...], window_len - window.shape[0], axis=0)
        window = np.concatenate([window, pad], axis=0)
    return window


def roll_nominal_from_qmean(q_mean: np.ndarray, step_advance: int, goal_state: np.ndarray) -> np.ndarray:
    """
    Build the next nominal trajectory by advancing along the planned mean and keeping the horizon length fixed.
    """
    horizon = q_mean.shape[0]
    tail = q_mean[step_advance:]
    if tail.shape[0] < horizon:
        pad = np.repeat(tail[-1:, ...], horizon - tail.shape[0], axis=0)
        tail = np.concatenate([tail, pad], axis=0)

    rolled = tail[:horizon]
    rolled[-1] = goal_state
    return rolled


def build_model(X_data, Y_data, dynamics, obstacle_centers_window, obstacle_radii_window):
    kernel = GaussMarkovKernel(
        X_data=X_data,
        dynamics=dynamics,
        anchor_vars=tf.constant([1e-5, 1e-5, 1e-3, 1e-3], dtype=default_float())
    )

    likelihood = PlanningLikelihood(
        dof=dynamics.dof,
        desired_nominal=Y_data,
        obstacle_centers_over_time=obstacle_centers_window,
        obstacle_radii_over_time=obstacle_radii_window,
        support_times=X_data,
        grid_size=10.0,
        epsilon=0.1,
        sigma_obs=0.02,
        sigma_nominal=0.8,
    )

    posterior = GaussMarkovPosterior(
        X_data=X_data,
        dynamics=dynamics
    )

    model = MarkovVGP(
        data=(X_data, Y_data),
        kernel=kernel,
        likelihood=likelihood,
        posterior=posterior,
        num_latent_gps=2 * dynamics.dof,
    )
    return model, kernel, posterior, likelihood


def run_training(model, num_steps: int, nsamples: int = 8, lr: float = 0.01):
    optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.8, beta_2=0.95)
    closure = model.training_loss_closure()

    step_iterator = tqdm(range(num_steps))
    start_ts = time.perf_counter()
    for step in step_iterator:
        loss = optimization_step(model, closure, optimizer, nsamples=nsamples)
        step_iterator.set_postfix_str(f"ELBO: {-loss:.3e}")

        # if step % 10 == 0:
        #     loss_v = float(loss.numpy())
        #     elbo_v = float(elbo.numpy())
        #     kl_v = float(KL.numpy())
        #     ve_v = float(sum_var_exp.numpy())
        #     print(
        #         f"[step {step:04d}] "
        #         f"loss={loss_v:.6e}  "
        #         f"ELBO={elbo_v:.6e}  "
        #         f"sum_var_exp={ve_v:.6e}  "
        #         f"KL={kl_v:.6e}"
        #     )
    elapsed = time.perf_counter() - start_ts
    print(f"Training wall-clock: {elapsed:.2f}s")


if __name__ == "__main__":
    # system's degrees of freedom
    dof = 2

    # start and goal positions
    start = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    goal = np.array([10.0, 10.0, 0.0, 0.0], dtype=np.float64)

    # time grid
    t0, tN = 0.0, 1.0
    num_interior_points = 9
    t = np.linspace(t0, tN, num_interior_points + 2)
    X_data = t.reshape(-1, 1).astype(np.float64)
    horizon_len = X_data.shape[0]

    # MPC settings
    execute_horizon = 2  # number of states to roll out before replanning
    mpc_iters = 1
    total_obstacle_steps = mpc_iters * horizon_len

    dynamics = ConstantVelocityModel(
        dof=dof,
        acceleration_noise=5.0,
    )

    # obstacle trajectory: moves left with a small drift; discrete samples only
    dt = float(t[1] - t[0])
    obstacle_centers_traj, obstacle_radii_traj = build_obstacle_trajectory(
        total_steps=total_obstacle_steps,
        center0=(5.0, 6.0),
        velocity=(0.00 * dt, 0.0 * dt),
        radius=1.5,
    )

    # initial nominal
    current_start = start
    Y_data = dynamics.initate_traj(
        times=X_data,
        start=current_start,
        goal=goal
    )

    prev_q_mean = None
    prev_L_mid = None
    prev_L_sub = None

    # main MPC loop
    obstacle_index = 0
    for mpc_iter in range(mpc_iters):
        obstacle_start_idx = obstacle_index
        centers_window = slice_window(obstacle_centers_traj, obstacle_start_idx, horizon_len)
        radii_window = slice_window(obstacle_radii_traj, obstacle_start_idx, horizon_len)

        model, kernel, posterior, likelihood = build_model(
            X_data=X_data,
            Y_data=Y_data,
            dynamics=dynamics,
            obstacle_centers_window=centers_window,
            obstacle_radii_window=radii_window,
        )

        # warm start from previous variational parameters if available
        if prev_q_mean is not None:
            model.q_mean_free.assign(prev_q_mean[1:-1])
            model.L_mid_diags_raw.assign(prev_L_mid)
            model.L_sub_raw.assign(prev_L_sub)

        run_training(model, num_steps=50, nsamples=8, lr=0.02)

        planned_q_mean = model.q_mean.numpy()
        prev_q_mean = planned_q_mean
        prev_L_mid = model.L_mid_diags_raw.numpy()
        prev_L_sub = model.L_sub_raw.numpy()

        # update nominal to match q_mean and advance along the planned path
        step_advance = min(execute_horizon, horizon_len - 1)
        Y_data = roll_nominal_from_qmean(planned_q_mean, step_advance=step_advance, goal_state=goal)
        current_start = Y_data[0]
        obstacle_index += step_advance

        print(f"[MPC iter {mpc_iter}] advanced start to {current_start}")

        # early exit if we have effectively reached the goal
        if np.allclose(current_start, goal, atol=1e-3):
            break

    # visualize the final plan with the last obstacle window
    plot_mean_and_obstacle(
        X_query=X_data,
        K_prior=kernel.K,
        model=model,
        posterior=posterior,
        nominal=prev_q_mean,
        obstacle_centers_over_time=centers_window,
        obstacle_radii_over_time=radii_window,
        epsilon=float(likelihood.epsilon),
        grid_size=float(likelihood.grid_size),
    )
