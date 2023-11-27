import numpy as np

from .field import Field

class Displacement(Field):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"Displacement(\n{self.data}\n)"