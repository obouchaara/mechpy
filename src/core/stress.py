import numpy as np
import sympy as sp

from tensor import SymmetricThreeByThreeTensor


class StressTensor(SymmetricThreeByThreeTensor):
    def __init__(self, data):
        # Call the constructor of the base class
        super().__init__(data)

    def pressure(self):
        # Calculate pressure from the trace of the stress tensor
        return np.trace(self.data) / 3.0
