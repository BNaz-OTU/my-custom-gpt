import numpy as np
import math
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)

        def custom_func(num): 
            return 1 / (1 + (math.e ** (-num)))
        
        vectorized_function = np.vectorize(custom_func)

        answer = vectorized_function(z)

        return np.round(answer, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z)