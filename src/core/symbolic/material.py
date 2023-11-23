import sympy as sp

from .tensor import SymbolicSixBySixTensor


class SymbolicAnisotropicMaterial:
    def __init__(self):
        pass

    def __repr__(self):
        return f"SymbolicAnisotropicMaterial()"


class SymbolicComplianceTensor(SymbolicSixBySixTensor):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"SymbolicComplianceTensor(\n{self.data}\n)"


class SymbolicStiffnessTensor(SymbolicSixBySixTensor):
    def __init__(self, data):
        super().__init__(data)

    def __repr__(self):
        return f"SymbolicStiffnessTensor(\n{self.data}\n)"


class SymbolicIsotropicMaterial(SymbolicAnisotropicMaterial):
    pass
