import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow

from vgpmp.prior import MaternPositionPrior
from vgpmp.likelihood import PlanningLikelihood
from vgpmp.models import VGPPlanner

DTYPE = tf.float64
gpflow.config.set_default_float(DTYPE)
gpflow.config.set_default_jitter(1e-5)

tf.debugging.enable_check_numerics()  
tf.config.run_functions_eagerly(True)


if __name__=="__main__":
    dof = 2
    t0  = 0.0                   # start time
    tN  = 1.0                   # end time
    num_interior = 9            # number of interior support points
    q_start = np.array([0.0, 0.0], dtype=np.float64)
    q_goal  = np.array([10.0, 10.0], dtype=np.float64)

    # Times excluding start and end
    t_interior = np.linspace(t0, tN, num_interior + 2)[1:-1]    # shape (num_interior,)
    X_in = t_interior.reshape(-1, 1).astype(np.float64)         # shape (K,1)

    # Dummy Y_in if your likelihood ignores Y
    Y_in = np.zeros((num_interior, dof), dtype=np.float64)      # shape (K, dof)

    X_ep = np.array([[t0], [tN]], dtype=np.float64)             # shape (2,1)
    Y_ep = np.stack([q_start, q_goal], axis=0)                  # shape (2,dof)

    # initialize posterior
    q_mu = tf.constant(
        [
            [1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0],
            [1.0, 2.0, 4.0, 4.5, 5.0, 6.0, 8.0, 8.5, 9.0],
        ],
        dtype=DTYPE
    )
    
    prior = MaternPositionPrior(dof, M=num_interior, t_min=t0, t_max=tN, matern_family="52")
    planning_lik = PlanningLikelihood(latent_dim=dof, obs_scale=2.0, temperature=0.8)
    model = VGPPlanner(prior, planning_lik, tf.transpose(q_mu, [1, 0]))

    data = {
        "interior": (X_in, Y_in),
        "endpoints": (X_ep, Y_ep),
    }
    loss_fn = model.training_loss_closure(data, compile=True)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.1)

    adam_vars = model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.01)

    @tf.function
    def optimization_step(data):
        tf.debugging.assert_all_finite(model.q_sqrt, "q_sqrt has NaNs before update")
        loss_fn = lambda: -model.elbo(data)
        # natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)
        tf.debugging.assert_all_finite(model.q_sqrt, "q_sqrt has NaNs after update")

    num_iters = 100
    for _ in range(num_iters):
        optimization_step(data)

    X_dense = np.linspace(t0, tN, 100, dtype=np.float64).reshape(-1,1)
    plot_mean_and_obstacle(model, X_dense, tf.transpose(q_mu, [1, 0]))