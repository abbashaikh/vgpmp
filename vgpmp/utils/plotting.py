import matplotlib as plt
import numpy as np

def plot_mean_and_obstacle(model, X, mu_0):
    # Model mean
    f_mean, f_var = model.predict_f(X, full_cov=False, full_output_cov=False)
    mean_pos = f_mean.numpy()
    var_pos = f_var.numpy()
    std_pos  = np.sqrt(np.maximum(var_pos, 0.0))

    mu_0 = mu_0.numpy()

    # Obstacle params
    cx, cy = model.likelihood.center.numpy()
    r = float(model.likelihood.radius.numpy())
    eps = float(model.likelihood.epsilon.numpy())
    L = float(model.likelihood.grid_size.numpy())

    fig, ax = plt.subplots()
    ax.plot(mean_pos[:, 0], mean_pos[:, 1], "-o", label="mean trajectory", color="C0")
    ax.plot(mu_0[:, 0], mu_0[:, 1], "x", label="initial trajectory", color="k")

    scale_x = std_pos[:, 0]
    lb_x = (mean_pos[:, 0] - 2*scale_x)
    ub_x = (mean_pos[:, 0] + 2*scale_x)
    scale_y = std_pos[:, 1]
    lb_y = (mean_pos[:, 1] - 2*scale_y)
    ub_y = (mean_pos[:, 1] + 2*scale_y)
    plt.fill_between(lb_x + lb_y, ub_x + ub_y, color="grey", alpha=0.1)

    # Obstacle circle and (optional) safety buffer R+epsilon
    circle = plt.Circle((cx, cy), r, color="red", alpha=0.25, label="obstacle")
    buffer = plt.Circle((cx, cy), r + eps, color="red", alpha=0.1, fill=False, linestyle="--", label="radius + epsilon")
    ax.add_patch(circle)
    ax.add_patch(buffer)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Mean trajectory and circular obstacle")
    plt.show()