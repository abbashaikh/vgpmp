from tqdm import tqdm

import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_float

from vgpmp.dynamics.constant_velocity import ConstantVelocityModel
from vgpmp.kernels import GaussMarkovKernel
from vgpmp.likelihood import PlanningLikelihood
from vgpmp.posterior import GaussMarkovPosterior
from vgpmp.models.markov_vgp import MarkovVGP
from vgpmp.utils.plotting import plot_mean_and_obstacle, visualize_initial_traj

"""
Notation:
    D -> total number of states
    N -> number of time segments
"""

gpflow.config.set_default_float(tf.float64)
gpflow.config.set_default_jitter(1e-6)


# TODO: implement natural gradients

@tf.function
def optimization_step(model, closure, optimizer, nsamples: int = 8):
    with tf.GradientTape() as tape:
        loss = closure()
        sum_var_exp, KL, elbo = model.elbo_terms(nsamples=nsamples)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, sum_var_exp, KL, elbo


if __name__=="__main__":
    # system's degrees of freedom
    dof = 2

    '''Train Data'''
    # start and goal positions
    start = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    goal  = np.array([10.0, 10.0, 0.0, 0.0], dtype=np.float64)
    # start and end time
    t0, tN  = 0.0, 1.0
    # number of interior support points (K)
    num_interior_points = 9
    # times
    t = np.linspace(t0, tN, num_interior_points + 2)
    X_data = t.reshape(-1, 1).astype(np.float64)         # shape (N+1,1)

    '''Test Data'''
    num_query_points = 20
    X_test = np.linspace(t0, tN, num_query_points, dtype=np.float64).reshape(-1,1)

    '''Model'''
    dynamics = ConstantVelocityModel(
        dof=dof,
        acceleration_noise=5.0,
    )

    kernel = GaussMarkovKernel(
        X_data=X_data,
        dynamics=dynamics,
        anchor_vars=tf.constant([1e-5, 1e-5, 1e-3, 1e-3], dtype=default_float())
    )

    Y_data = dynamics.initate_traj(
        times=X_data,
        start=start,
        goal=goal
    )
    # visualize_initial_traj(
    #     X_data,
    #     dynamics,
    #     mean,
    # )

    # Build likelihood for circular obstacles
    obstacles = [
        ((7.0, 7.0), 1.0),
        # ((3.0, 4.0), 1.0),
    ]
    likelihood = PlanningLikelihood(
        dof=dof,
        desired_nominal=Y_data,
        obstacles=obstacles,
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
        num_latent_gps=2*dof
    )

    #TODO: set trainable parameters and assign appropriate priors to them

    '''Training loop'''
    num_steps = 250
    optimizer = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.8, beta_2=0.95)
    closure = model.training_loss_closure()

    step_iterator = tqdm(range(num_steps))
    for step in step_iterator:
        loss, sum_var_exp, KL, elbo = optimization_step(model, closure, optimizer, nsamples=8)

        # tqdm postfix every step (lightweight)
        step_iterator.set_postfix_str(f"ELBO: {-loss:.3e}")

        # detailed print every 10 steps
        if step % 10 == 0:
            # Convert to python floats for clean printing
            loss_v = float(loss.numpy())
            elbo_v = float(elbo.numpy())
            kl_v = float(KL.numpy())
            ve_v = float(sum_var_exp.numpy())

            print(
                f"[step {step:04d}] "
                f"loss={loss_v:.6e}  "
                f"ELBO={elbo_v:.6e}  "
                f"sum_var_exp={ve_v:.6e}  "
                f"KL={kl_v:.6e}"
            )

    # opt = tf.keras.optimizers.Adam(learning_rate=1e-2)
    # for it in range(1000):
    #     loss_val = step(model, nsamples=32)
    #     if it % 100 == 0:
    #         tf.print("iter", it, "negELBO", loss_val)

    # scipy_opt = gf.optimizers.Scipy()

    # scipy_opt.minimize(
    #     model.objective_closure(nsamples=128),
    #     variables=model.trainable_variables,
    #     options=dict(maxiter=200)
    # )

    '''Output'''
    ## Posterior mean and variance plot
    plot_mean_and_obstacle(
        X_query=X_test,
        K_prior=kernel.K,
        model=model,
        posterior=posterior,
        nominal=Y_data,
        obstacles=obstacles,
        epsilon=likelihood.epsilon,
        grid_size=likelihood.grid_size
    )





