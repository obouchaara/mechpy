import sympy as sp

from .field import SymbolicVectorField

class SymbolicVolumeForce(SymbolicVectorField):
    def __init__(self, coord_system, data, field_params=None, symbols_validation=True):
        super().__init__(coord_system, data, field_params, symbols_validation)