import sympy as sp

from .field import SymbolicVectorField

class SymbolicDisplacement(SymbolicVectorField):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"SymbolicDisplacement(\n{self.data}\n)"