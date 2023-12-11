import sympy as sp


class SymbolicCoordSystem:
    def __init__(self, origin, basis_symbols):
        self.origin = origin
        self.basis_symbols = basis_symbols

    def __repr__(self):
        return f"{self.__class__.__name__}(origin={self.origin}, basis_symbols={self.basis_symbols})"


class SymbolicCartesianCoordSystem(SymbolicCoordSystem):
    def __init__(self, basis_symbols=None):
        origin = sp.ImmutableDenseNDimArray([0, 0, 0])
        basis_symbols = basis_symbols or [sp.symbols(_) for _ in ["x", "y", "z"]]
        super().__init__(origin, basis_symbols)


class SymbolicCylindricalCoordSystem(SymbolicCoordSystem):
    def __init__(self, basis_symbols=None):
        origin = sp.ImmutableDenseNDimArray([0, 0, 0])
        basis_symbols = basis_symbols or [sp.symbols(_) for _ in ["r", "theta", "z"]]
        super().__init__(origin, basis_symbols)

    def to_cartesian(self):
        r, theta, z = self.basis_symbols
        x = r * sp.cos(theta)
        y = r * sp.sin(theta)
        basis_symbols = sp.ImmutableDenseNDimArray([x, y, z])
        return SymbolicCartesianCoordSystem(basis_symbols=basis_symbols)

    def get_cartesian_coords(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("values must be a list or tuple of length 3")
        r, theta, z = self.basis_symbols
        value_dict = {r: values[0], theta: values[1], z: values[2]}
        cartesian_system = self.to_cartesian()
        cartesian_coords = cartesian_system.basis_symbols.subs(value_dict)
        return cartesian_coords


class SymbolicSphericalCoordSystem(SymbolicCoordSystem):
    def __init__(self, basis_symbols=None):
        origin = sp.ImmutableDenseNDimArray([0, 0, 0])
        basis_symbols = basis_symbols or [sp.symbols(_) for _ in ["r", "theta", "phi"]]
        super().__init__(origin, basis_symbols)

    def to_cartesian(self):
        r, theta, phi = self.basis_symbols
        x = r * sp.sin(phi) * sp.cos(theta)
        y = r * sp.sin(phi) * sp.sin(theta)
        z = r * sp.cos(phi)
        basis_symbols = sp.ImmutableDenseNDimArray([x, y, z])
        return SymbolicCartesianCoordSystem(basis_symbols=basis_symbols)
    
    def get_cartesian_coords(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("values must be a list or tuple of length 3")

        r, theta, phi = self.basis_symbols
        value_dict = {r: values[0], theta: values[1], phi: values[2]}

        cartesian_system = self.to_cartesian()
        cartesian_coords = cartesian_system.basis_symbols.subs(value_dict)

        return cartesian_coords
