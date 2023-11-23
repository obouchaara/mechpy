import sympy as sp

class SymbolicCoordSystem:
    def __init__(self, origin=None, basis_symbols=None):
        self.origin = origin or [0, 0, 0]
        self.basis_symbols = basis_symbols or [sp.symbols(f'x{i}') for i in range(3)]

    def __repr__(self):
        return f"SymbolicCoordSystem(origin={self.origin}, basis_symbols={self.basis_symbols})"

    def to_local(self, global_coords):
        return [sp.solve((global_coord - origin, basis_sym), basis_sym)[0] for basis_sym, global_coord, origin in zip(self.basis_symbols, global_coords, self.origin)]

    def to_global(self, local_coords):
        return [sum([coeff * local_coord for coeff, local_coord in zip(basis_sym.as_coefficients_dict(), local_coords)]) + origin for basis_sym, origin in zip(self.basis_symbols, self.origin)]
