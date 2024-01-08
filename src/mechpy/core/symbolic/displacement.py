import sympy as sp

from .strain import SymbolicStrainTensor
from .coord import SymbolicCartesianCoordSystem
from .field import SymbolicVectorField


class SymbolicDisplacement(SymbolicVectorField):
    def __init__(self, data, coord_system=None, field_params=None):
        super().__init__(data, coord_system, field_params)

    def __repr__(self):
        return f"SymbolicDisplacement(\n{self.data}\n)"

    def strain_tensor(self, coord_system=None) -> SymbolicStrainTensor:
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        x1, x2, x3 = coord_system.basis_symbols
        u, v, w = self.data
        e_11 = sp.diff(u, x1)
        e_22 = sp.diff(v, x2)
        e_33 = sp.diff(w, x3)
        e_12 = sp.diff(u, x2) + sp.diff(v, x1)
        e_23 = sp.diff(v, x3) + sp.diff(w, x2)
        e_31 = sp.diff(w, x1) + sp.diff(u, x3)
        components = [e_11, e_22, e_33, e_12, e_23, e_31]
        strain_tensor = SymbolicStrainTensor.from_list(components, notation=2)
        return strain_tensor
