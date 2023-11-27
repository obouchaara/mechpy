import numpy as np

class Field:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError("Input data must be a NumPy array")

    def __repr__(self):
        return f"Field(\n{self.data}\n)"

    def gradient(self):
        return np.gradient(self.data)