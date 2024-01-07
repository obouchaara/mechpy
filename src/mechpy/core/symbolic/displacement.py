import sympy as sp

from .field import SymbolicVectorField


class SymbolicDisplacement(SymbolicVectorField):
    def __init__(self, data, coord_system=None, field_params=None):
        super().__init__(data, coord_system, field_params)

    def __repr__(self):
        return f"SymbolicDisplacement(\n{self.data}\n)"
