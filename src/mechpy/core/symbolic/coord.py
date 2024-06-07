from typing import Tuple, Union
import sympy as sp


class SymbolicCoordSystem:
    def __init__(
        self,
        origin: Tuple[sp.Number],
        basis: Tuple[Union[sp.Symbol, sp.Expr]],
    ):
        if not isinstance(origin, tuple) or not all(
            isinstance(coord, sp.Number) for coord in origin
        ):
            raise ValueError("origin must be a tuple of sympy.Number")
        if not isinstance(basis, tuple) or not all(
            isinstance(symbol, (sp.Symbol, sp.Expr)) for symbol in basis
        ):
            raise ValueError(
                "basis must be a tuple of sympy.Symbol or sympy.Expr instances"
            )
        if len(origin) != len(basis):
            raise ValueError("origin and basis must have the same length")

        self.origin = origin
        self.basis = basis

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(origin={self.origin}, basis={self.basis})"

    @classmethod
    def auto_detect(cls, data):
        # Implement logic for auto-detecting coordinate systems based on input data
        raise NotImplementedError

    @classmethod
    def coord_subs(cls, data: Tuple[Union[sp.Symbol, sp.Expr]], dict: dict):
        items = [item.subs(dict) for item in data]
        return tuple(items)

    @classmethod
    def coord_eval(cls, data: Tuple[Union[sp.Symbol, sp.Expr]], subs=None):
        items = [item.evalf(subs=subs) for item in data]
        return tuple(items)


class SymbolicCartesianCoordSystem(SymbolicCoordSystem):
    def __init__(
        self,
        origin: Tuple[sp.Number] = None,
        basis: Tuple[Union[sp.Symbol, sp.Expr]] = None,
    ):
        """
        Initialize a symbolic Cartesian coordinate system.
        :param origin: A tuple representing the origin. Defaults (0, 0, 0).
        :param basis: Optional tuple of basis symbols. Defaults to ('x', 'y', 'z').
        """
        origin = origin or (sp.Number(0), sp.Number(0), sp.Number(0))
        basis = basis or sp.symbols("x y z")
        if len(basis) != 3:
            raise ValueError("basis must have a length of 3")
        super().__init__(origin=origin, basis=basis)

    def get_basis_cylindrical_exprs(self, cylindrical_basis=None) -> dict:
        """
        Get the expressions for cylindrical basis in terms of Cartesian basis.
        :return: A dictionary with cylindrical basis expressions.
        """
        x, y, z = self.basis
        r, theta, z = cylindrical_basis or sp.symbols("r theta z")
        return {
            r: sp.sqrt(x**2 + y**2),
            theta: sp.atan2(y, x),
            z: z,
        }

    def get_basis_spherical_exprs(self, spherical_basis=None) -> dict:
        """
        Get the expressions for spherical basis in terms of Cartesian basis.
        :return: A dictionary with spherical basis expressions.
        """
        x, y, z = self.basis
        r, theta, phi = spherical_basis or sp.symbols("r theta phi")
        r_expr = sp.sqrt(x**2 + y**2 + z**2)
        return {
            r: r_expr,
            theta: sp.atan2(y, x),
            phi: sp.acos(z / r_expr),
        }

    def to_cylindrical(self):
        """
        Convert to cylindrical coordinate system.
        :return: An instance of SymbolicCylindricalCoordSystem.
        """
        expr_dict = self.get_basis_cylindrical_exprs()
        basis = tuple(expr_dict.values())
        return SymbolicCylindricalCoordSystem(origin=self.origin, basis=basis)

    def to_spherical(self):
        """
        Convert to spherical coordinate system.
        :return: An instance of SymbolicSphericalCoordSystem.
        """
        expr_dict = self.get_basis_spherical_exprs()
        basis = tuple(expr_dict.values())
        return SymbolicSphericalCoordSystem(origin=self.origin, basis=basis)

    def get_cylindrical_coord(self, values: tuple) -> tuple:
        """
        Convert Cartesian coordinates to cylindrical coordinates.
        :param values: A tuple of Cartesian coordinates (x, y, z).
        :return: An tuple of cylindrical coordinates (r, theta, z).
        """
        if len(values) != 3:
            raise ValueError("values must be a tuple of length 3.")
        cylindrical_system = self.to_cylindrical()
        substitutions = dict(zip(self.basis, values))
        cylindrical_coords = SymbolicCoordSystem.coord_subs(
            cylindrical_system.basis,
            substitutions,
        )
        return cylindrical_coords

    def get_spherical_coord(self, values: tuple) -> tuple:
        """
        Convert Cartesian coordinates to spherical coordinates.
        :param values: A tuple of Cartesian coordinates (x, y, z).
        :return: An tuple of spherical coordinates (r, theta, phi).
        """
        if len(values) != 3:
            raise ValueError("values must be a tuple of length 3.")
        spherical_system = self.to_spherical()
        substitutions = dict(zip(self.basis, values))
        spherical_coords = SymbolicCoordSystem.coord_subs(
            spherical_system.basis,
            substitutions,
        )
        return spherical_coords


class SymbolicCylindricalCoordSystem(SymbolicCoordSystem):
    def __init__(
        self,
        origin: Tuple[Union[int, float]] = None,
        basis: Tuple[Union[sp.Symbol, sp.Expr]] = None,
    ):
        origin = origin or (sp.Number(0), sp.Number(0), sp.Number(0))
        basis = basis or sp.symbols("r theta z")
        if len(basis) != 3:
            raise ValueError("basis must have a length of 3")
        super().__init__(origin=origin, basis=basis)

    def get_basis_cartesian_exprs(self, cartesian_basis=None) -> dict:
        r, theta, z = self.basis
        x, y, z_cart = cartesian_basis or sp.symbols("x y z")
        return {
            x: r * sp.cos(theta),
            y: r * sp.sin(theta),
            z_cart: z,
        }

    def to_cartesian(self):
        expr_dict = self.get_basis_cartesian_exprs()
        basis = tuple(expr_dict.values())
        return SymbolicCartesianCoordSystem(origin=self.origin, basis=basis)

    def get_cartesian_coords(self, values):
        if not isinstance(values, tuple) or len(values) != 3:
            raise ValueError("values must be a tuple of length 3")
        cartesian_system = self.to_cartesian()
        value_dict = dict(zip(self.basis, values))
        cartesian_coords = [expr.subs(value_dict) for expr in cartesian_system.basis]
        return tuple(cartesian_coords)


class SymbolicSphericalCoordSystem(SymbolicCoordSystem):
    def __init__(
        self,
        origin: Tuple[Union[int, float]] = None,
        basis: Tuple[Union[sp.Symbol, sp.Expr]] = None,
    ):
        origin = origin or (sp.Number(0), sp.Number(0), sp.Number(0))
        basis = basis or sp.symbols("r theta z")
        if len(basis) != 3:
            raise ValueError("basis must be a tuple of 3.")
        super().__init__(origin=origin, basis=basis)

    def get_basis_cartesian_exprs(self, cartesian_basis=None) -> dict:
        r, theta, phi = self.basis
        cartesian_basis = cartesian_basis or sp.symbols("x y z")
        x, y, z_cart = cartesian_basis
        return {
            x: r * sp.sin(phi) * sp.cos(theta),
            y: r * sp.sin(phi) * sp.sin(theta),
            z_cart: r * sp.cos(phi),
        }

    def to_cartesian(self):
        expr_dict = self.get_basis_cartesian_exprs()
        basis = tuple(expr_dict.values())
        return SymbolicCartesianCoordSystem(origin=self.origin, basis=basis)

    def get_cartesian_coords(self, values):
        if not isinstance(values, tuple) or len(values) != 3:
            raise ValueError("values must be a tuple of length 3")
        cartesian_system = self.to_cartesian()
        value_dict = dict(zip(self.basis, values))
        cartesian_coords = [expr.subs(value_dict) for expr in cartesian_system.basis]
        return tuple(cartesian_coords)
