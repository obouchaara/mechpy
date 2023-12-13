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

    def to_cylindrical(self):
        x, y, z = self.basis_symbols
        r = sp.sqrt(x**2 + y**2)
        theta = sp.atan2(y, x)
        basis_symbols = sp.ImmutableDenseNDimArray([r, theta, z])
        return SymbolicCylindricalCoordSystem(basis_symbols=basis_symbols)

    def to_spherical(self):
        x, y, z = self.basis_symbols
        r = sp.sqrt(x**2 + y**2 + z**2)
        theta = sp.atan2(y, x)
        phi = sp.acos(z / r)
        basis_symbols = sp.ImmutableDenseNDimArray([r, theta, phi])
        return SymbolicSphericalCoordSystem(basis_symbols=basis_symbols)

    def get_cylindrical_coord(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("values must be a list or tuple of length 3")

        x, y, z = self.basis_symbols
        value_dict = {x: values[0], y: values[1], z: values[2]}
        cylindrical_system = self.to_cylindrical()
        cylindrical_coords = cylindrical_system.basis_symbols.subs(value_dict)

        return cylindrical_coords

    def get_spherical_coord(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("values must be a list or tuple of length 3")

        x, y, z = self.basis_symbols
        value_dict = {x: values[0], y: values[1], z: values[2]}
        spherical_system = self.to_spherical()
        spherical_coords = spherical_system.basis_symbols.subs(value_dict)

        return spherical_coords


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
