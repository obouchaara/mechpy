import sympy as sp


class SymbolicCoordSystem:
    def __init__(self, origin, basis_symbols):
        self.origin = origin
        self.basis_symbols = basis_symbols

    def __repr__(self):
        return f"{self.__class__.__name__}(origin={self.origin}, basis_symbols={self.basis_symbols})"

    @classmethod
    def auto_detect(cls, data):
        pass


class SymbolicCartesianCoordSystem(SymbolicCoordSystem):
    def __init__(self, basis_symbols=None):
        origin = (0, 0, 0)
        basis_symbols = basis_symbols or sp.symbols("x y z")
        super().__init__(origin, basis_symbols)

    def get_basis_cylindrical_exprs(self) -> dict:
        x, y, z = self.basis_symbols
        r_expr = sp.sqrt(x**2 + y**2)
        theta_expr = sp.atan2(y, x)
        z_expr = z  # z coordinate remains the same
        return {"r": r_expr, "theta": theta_expr, "z": z_expr}

    def get_basis_spherical_exprs(self) -> dict:
        x, y, z = self.basis_symbols
        r_expr = sp.sqrt(x**2 + y**2 + z**2)
        theta_expr = sp.atan2(y, x)
        phi_expr = sp.acos(z / r_expr)
        return {"r": r_expr, "theta": theta_expr, "phi": phi_expr}

    def to_cylindrical(self):
        expr_dict = self.get_basis_cylindrical_exprs()
        basis_symbols = sp.ImmutableDenseNDimArray(list(expr_dict.values()))
        return SymbolicCylindricalCoordSystem(basis_symbols=basis_symbols)

    def to_spherical(self):
        expr_dict = self.get_basis_spherical_exprs()
        basis_symbols = sp.ImmutableDenseNDimArray(list(expr_dict.values()))
        return SymbolicSphericalCoordSystem(basis_symbols=basis_symbols)

    def get_cylindrical_coord(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("values must be a list or tuple of length 3")
        cylindrical_system = self.to_cylindrical()
        value_dict = dict(zip(self.basis_symbols, values))
        cylindrical_coords = cylindrical_system.basis_symbols.subs(value_dict)
        return cylindrical_coords

    def get_spherical_coord(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("values must be a list or tuple of length 3")
        spherical_system = self.to_spherical()
        value_dict = dict(zip(self.basis_symbols, values))
        spherical_coords = spherical_system.basis_symbols.subs(value_dict)
        return spherical_coords


class SymbolicCylindricalCoordSystem(SymbolicCoordSystem):
    def __init__(self, basis_symbols=None):
        origin = (0, 0, 0)
        basis_symbols = basis_symbols or sp.symbols("r theta z")
        super().__init__(origin, basis_symbols)

    def get_basis_cartesian_exprs(self, cartesian_basis_symbols=None) -> dict:
        if not cartesian_basis_symbols:
            cartesian_basis_symbols = sp.symbols("x y z")
        r, theta, z_cyl = self.basis_symbols
        x, y, z_cart = cartesian_basis_symbols
        r_expr = sp.sqrt(x**2 + y**2)
        theta_expr = sp.atan2(y, x)
        return {r: r_expr, theta: theta_expr, z_cyl: z_cart}

    def to_cartesian(self):
        expr_dict = self.get_basis_cartesian_exprs()
        basis_symbols = sp.ImmutableDenseNDimArray(list(expr_dict.values()))
        return SymbolicCartesianCoordSystem(basis_symbols=basis_symbols)

    def get_cartesian_coords(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("values must be a list or tuple of length 3")
        cartesian_system = self.to_cartesian()
        value_dict = dict(zip(self.basis_symbols, values))
        cartesian_coords = cartesian_system.basis_symbols.subs(value_dict)
        return cartesian_coords


class SymbolicSphericalCoordSystem(SymbolicCoordSystem):
    def __init__(self, basis_symbols=None):
        origin = (0, 0, 0)
        basis_symbols = basis_symbols or sp.symbols("r theta phi")
        super().__init__(origin, basis_symbols)

    def get_basis_cartesian_exprs(self, cartesian_basis_symbols=None) -> dict:
        if not cartesian_basis_symbols:
            cartesian_basis_symbols = sp.symbols("x y z")
        r, theta, phi = self.basis_symbols
        x, y, z = cartesian_basis_symbols
        r_expr = sp.sqrt(x**2 + y**2 + z**2)
        theta_expr = sp.atan2(y, x)
        phi_expr = sp.acos(z / r_expr)

        return {r: r_expr, theta: theta_expr, phi: phi_expr}

    def to_cartesian(self):
        expr_dict = self.get_basis_cartesian_exprs()
        basis_symbols = sp.ImmutableDenseNDimArray(list(expr_dict.values()))
        return SymbolicCartesianCoordSystem(basis_symbols=basis_symbols)

    def get_cartesian_coords(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("values must be a list or tuple of length 3")
        cartesian_system = self.to_cartesian()
        value_dict = dict(zip(self.basis_symbols, values))
        cartesian_coords = cartesian_system.basis_symbols.subs(value_dict)
        return cartesian_coords
