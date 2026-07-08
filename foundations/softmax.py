import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z_shifted = z - max(z)
        sum = np.exp(z_shifted).sum()

        for i, value in enumerate(z_shifted):
            z_shifted[i] = np.exp(value) / sum


        return np.round(z_shifted, 4)
