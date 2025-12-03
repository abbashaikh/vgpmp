from typing import Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from gpflow import Parameter

TensorType = Union[np.ndarray ,tf.Tensor, tf.Variable, Parameter]
InputData = Union[TensorType]
OutputData = Union[TensorType]
RegressionData = Tuple[InputData, OutputData]