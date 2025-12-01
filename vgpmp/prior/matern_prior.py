# prior.py
import gpflow
import numpy as np
import tensorflow as tf

class MaternPositionPrior:
    """
    Multioutput GP prior over joint positions q(t) ∈ R^{dof}.
    Input: t ∈ R (shape [N,1]). Output: shape [N, dof].
    """
    def __init__(
        self,
        dof: int,
        M: int,
        t_min: float,
        t_max: float,
        matern_family: str = "52"
    ):
        self.dof = dof
        self.t_min, self.t_max = t_min, t_max

        # kernel: one independent Matérn per joint (shared input dimension = time)
        base_kernels = []
        for _ in range(dof):
            if matern_family == "32":
                k = gpflow.kernels.Matern32(lengthscales=0.3, variance=1.0)
            elif matern_family == "12":
                k = gpflow.kernels.Matern12(lengthscales=0.3, variance=1.0)
            else:
                k = gpflow.kernels.Matern52(lengthscales=0.3, variance=1.0)
            base_kernels.append(k)

        self.kernel = gpflow.kernels.SeparateIndependent(base_kernels)

        # shared inducing times Z (M x 1), replicated across outputs
        Z = np.linspace(t_min, t_max, M).reshape(-1, 1)
        iv = gpflow.inducing_variables.InducingPoints(Z.astype(np.float64))
        self.inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(iv)

        # Gaussian likelihood variance is for endpoint “observations” only (tiny noise)
        self.obs_likelihood = gpflow.likelihoods.Gaussian(variance=1e-5)

    def initial_Z(self):
        return self.inducing_variable.inducing_variables.Z  # (M,1)
