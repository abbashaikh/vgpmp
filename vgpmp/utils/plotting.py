import matplotlib as plt
from matplotlib.patches import Circle, Ellipse

import numpy as np
import tensorflow as tf


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


def plot_mean_and_obstacle(
        X_query,
        Kinv_cholesky,
        model,
        posterior,
        obstacle_center,
        obstacle_radius,
        epsilon,
        grid_size,
    ):
    # ----- Evaluate GP posterior at query points -----
    dof = posterior.dynamics.dof
    mean, variance = posterior.predict_f(X_query, model.q_mean, model.q_cov, Kinv_cholesky)

    mean = mean.numpy() if tf.is_tensor(mean) else np.asarray(mean)
    variance = variance.numpy() if tf.is_tensor(variance) else np.asarray(variance)
    
    x_mean = mean[:, 0]         # (M,)
    y_mean = mean[:, 1]         # (M,)

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(6, 6))
    # mean trajectory
    ax.plot(x_mean, y_mean, label="Mean trajectory")
    # covairance ellipses
    first_ellipse = True
    for i in range(0, len(x_mean), 2):
        cov = variance[i, :dof, :dof]       # (dof, dof)
        if not np.all(np.isfinite(cov)):
            continue
        _plot_cov_ellipse(
            (mean[i, 0], mean[i, 1]),
            cov,
            label="Covariance ellipse" if first else None,
        )
        first = False

    # Obstacle circle and (optional) safety buffer R+epsilon
    circle = Circle(obstacle_center, obstacle_radius, color="red", alpha=0.25, label="obstacle")
    buffer = Circle(
        obstacle_center,
        obstacle_radius + epsilon,
        fill=False,
        linestyle="--",
        alpha=0.7,
        color="red",
        label="Radius + epsilon",
    )
    ax.add_patch(circle)
    ax.add_patch(buffer)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.set_title("Posterior trajectory and circular obstacle")

    plt.show()

    return ax