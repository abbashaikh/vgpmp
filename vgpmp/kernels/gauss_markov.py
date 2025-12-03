from gpflow.kernels import Kernel

class GaussMarkovKernel(Kernel):
    def __init__(self, active_dims = None, name = None):
        super().__init__(active_dims, name)