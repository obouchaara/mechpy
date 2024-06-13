import sympy as sp

from .coord import SymbolicCartesianCoordSystem
from .field import SymbolicVectorField
from .strain import SymbolicStrainTensor


class SymbolicDisplacement(SymbolicVectorField):
    def __init__(self, coord_system, data, field_params=None, symbols_validation=True):
        super().__init__(coord_system, data, field_params, symbols_validation)

    def __repr__(self):
        return f"SymbolicDisplacement(\n{self.data}\n)"

    def strain_tensor(self, coord_system=None) -> SymbolicStrainTensor:
        if coord_system is None:
            coord_system = self.coord_system

        x1, x2, x3 = coord_system.basis
        u, v, w = self.data
        e_11 = sp.diff(u, x1)
        e_22 = sp.diff(v, x2)
        e_33 = sp.diff(w, x3)
        e_12 = sp.diff(u, x2) + sp.diff(v, x1)
        e_23 = sp.diff(v, x3) + sp.diff(w, x2)
        e_31 = sp.diff(w, x1) + sp.diff(u, x3)
        components = [e_11, e_22, e_33, e_12, e_23, e_31]
        strain_tensor = SymbolicStrainTensor.from_list(components, notation="voigt")
        return strain_tensor
