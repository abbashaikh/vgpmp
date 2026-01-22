import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

import numpy as np
import tensorflow as tf
from gpflow.config import default_float


def _plot_cov_ellipse(ax, center, cov, label=None):
    cov = 0.5 * (cov + cov.T)

    # eigenvalues
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, np.inf)
    # largest eigenvalue eigenvector
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # ellipse radi (2-std) and orientation
    r1 = 2.0 * np.sqrt(vals[0])
    r2 = 2.0 * np.sqrt(vals[1])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # plot ellipse
    e = Ellipse(
        xy=center,
        width=2.0 * r1,
        height=2.0 * r2,
        angle=angle,
        fill=False,
        lw=0.8,
        alpha=0.5,
        color="C0",
        label=label,
    )
    ax.add_patch(e)


def visualize_initial_traj(
    times,
    dynamics,
    mean: tf.Tensor
):
    times = tf.convert_to_tensor(times, dtype=default_float())
    dt = times[1:] - times[:-1]

    dof = dynamics.dof
    P = dynamics.state_dimension

    K0 = 1.0 * np.eye(P)

    Phi_list = dynamics.get_transition_matrices(dt[:, 0])
    Q_list   = dynamics.get_noise_matrices(dt[:, 0])

    covs = [K0]

    for k in range(len(Phi_list)):
        Phi = Phi_list[k].numpy()
        Q   = Q_list[k].numpy()
        K_k  = Phi @ covs[-1] @ Phi.T + Q
        covs.append(K_k)
    covs = np.stack(covs)  # (N,P,P)

    mean = mean.numpy()

    # ----- Position -----
    pos_mean = mean[:, 0:dof]   # 2D position
    pos_var  = covs[:, 0, 0]    # variance of position
    pos_std = np.sqrt(pos_var)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(pos_mean[:, 0], pos_mean[:, 1], linewidth=2, label="Mean trajectory")
    ax.scatter(pos_mean[0, 0],  pos_mean[0, 1],  c="k", s=30, zorder=4, label="Start")
    ax.scatter(pos_mean[-1, 0], pos_mean[-1, 1], c="k", s=30, zorder=4, marker="x", label="Goal")

    # Ellipses (position covariance block)
    first_ellipse = True
    for k in range(0, pos_mean.shape[0], 1):
        C_pos = covs[k, 0:2, 0:2]  # position covariance
        _plot_cov_ellipse(
            ax,
            center=pos_mean[k],
            cov=C_pos,
            label="2σ covariance" if first_ellipse else None,
        )
        first_ellipse = False

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D Initial Trajectory with 2σ Covariance Ellipses")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # ----- Velocity -----
    t = times[:, 0].numpy()
    vx_mean = mean[:, 2]
    vx_std  = np.sqrt(covs[:, 2, 2])

    plt.figure(figsize=(8, 4))
    plt.plot(t, vx_mean, lw=2, label="Mean $v_x$")
    plt.fill_between(
        t,
        vx_mean - 2 * vx_std,
        vx_mean + 2 * vx_std,
        alpha=0.3,
        label="±2σ"
    )
    plt.xlabel("Time")
    plt.ylabel("$v_x$")
    plt.title("Velocity $v_x(t)$ with 2σ Bounds")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def plot_mean_and_obstacle(
        X_query,
        K_prior,
        model,
        posterior,
        nominal,
        obstacles,
        epsilon,
        grid_size,
    ):
    # ----- Evaluate GP posterior at query points -----
    dof = posterior.dynamics.dof
    mean, variance = posterior.predict_f(X_query, model.q_mean, model.q_cov, K_prior)

    mean = mean.numpy() if tf.is_tensor(mean) else np.asarray(mean)
    variance = variance.numpy() if tf.is_tensor(variance) else np.asarray(variance)
    
    x_mean = mean[:, 0]         # (M,)
    y_mean = mean[:, 1]         # (M,)

    nominal = nominal.numpy() if tf.is_tensor(nominal) else np.asarray(nominal)
    x_nom = nominal[:, 0]
    y_nom = nominal[:, 1]

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(6, 6))

    # mean trajectory
    ax.plot(x_mean, y_mean, label="Mean trajectory")

    ax.plot(x_nom, y_nom, linestyle="--", linewidth=2, label="Nominal trajectory")

    # covairance ellipses
    first_ellipse = True
    for i in range(0, len(x_mean), 1):
        cov = variance[i, :dof, :dof]       # (dof, dof)
        if not np.all(np.isfinite(cov)):
            continue
        _plot_cov_ellipse(
            ax,
            (mean[i, 0], mean[i, 1]),
            cov,
            label="Covariance ellipse" if first_ellipse else None,
        )
        first_ellipse = False

    # Obstacle circle and (optional) safety buffer R+epsilon
    # obstacles + safety buffers
    first_obs = True
    first_buf = True
    for (center, radius) in obstacles:
        cx, cy = center

        circle = Circle(
            (cx, cy),
            radius,
            color="red",
            alpha=0.25,
            label="Obstacle" if first_obs else None,
        )
        buffer = Circle(
            (cx, cy),
            radius + float(epsilon),
            fill=False,
            linestyle="--",
            alpha=0.7,
            color="red",
            label="Radius + epsilon" if first_buf else None,
        )
        ax.add_patch(circle)
        ax.add_patch(buffer)

        first_obs = False
        first_buf = False

    ax.set_xlim(-5, grid_size+5)
    ax.set_ylim(-5, grid_size+5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.set_title("Posterior trajectory and circular obstacle")

    plt.show()

    return ax