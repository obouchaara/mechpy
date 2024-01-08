import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


from .coord import (
    SymbolicCoordSystem,
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)


class SymbolicField:
    def __init__(self, data, coord_system, field_params=None):
        if isinstance(coord_system, SymbolicCoordSystem):
            self.data = data
            self.coord_system = coord_system
            self.field_params = list(field_params or [])
            self.validate_field()
        else:
            raise ValueError("coord system must be a SymbolicCoordSystem")

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data},\n{self.coord_system.basis_symbols},\n{self.field_params})"

    def validate_field(self):
        self.validate_field_params()
        self.validate_basis_symbols()

    def validate_field_params(self):
        if self.field_params:
            field_param_symbols = set(self.field_params)
            basis_symbols = set(self.coord_system.basis_symbols)

            if not field_param_symbols.isdisjoint(basis_symbols):
                raise ValueError(
                    "Field parameters must not overlap with coordinate system basis symbols."
                )

    def validate_basis_symbols(self):
        """
        Validates that all free symbols in the field data are either part of the
        coordinate system's basis symbols or the field parameters.

        Raises a ValueError if there are any symbols in the field data that are
        not included in either the basis symbols of the coordinate system or the
        field parameters.

        This ensures that the field data is properly defined with respect to the
        coordinate system and any additional parameters.
        """
        # extract basis symbols function arg
        # extract data function arg
        
        basis_symbols = set(self.coord_system.basis_symbols)
        field_param_symbols = set(self.field_params)
        
        # to remove
        # Extracting the arguments of each function
        function_args = set()
        for element in self.data:
            for arg in element.args:
                if arg.is_Function:
                    function_args.update(arg.args)
        valid_symbols = basis_symbols.union(field_param_symbols).union(function_args)
        # to remove
        
        # valid_symbols = basis_symbols.union(field_param_symbols)

        free_symbols = (
            self.data.free_symbols
            if isinstance(self.data, sp.Expr)
            else set().union(*[element.free_symbols for element in self.data])
        )

        # Exclude numerical symbols from the free symbols set
        free_symbols = {sym for sym in free_symbols if not isinstance(sym, sp.Number)}


        invalid_symbols = free_symbols - valid_symbols
        if invalid_symbols:
            raise ValueError(
                "The field data contains symbols not in the basis or field parameters: "
                + ", ".join(str(symbol) for symbol in invalid_symbols)
            )

    def subs_field_params(self, param_values):
        """
        Substitute the provided field parameters with specific values, and
        remove them from self.field_params. Raise an error if a parameter in
        param_values is not in self.field_params.

        :param param_values: A dictionary mapping parameters to their values.
        :return: None. The method updates self.data and self.field_params in place.
        """
        if not isinstance(param_values, dict):
            raise TypeError("param_values must be a dictionary")

        # Perform the substitution for provided parameters
        for param, value in param_values.items():
            if param in self.field_params:
                self.data = self.data.subs(param, value)
                self.field_params.remove(param)
            else:
                raise ValueError(f"Parameter {param} not found in field parameters")

    def to_cartesian(self):
        """
        Converts the scalar field from its current coordinate system
        (cylindrical or spherical) to the Cartesian coordinate system.

        Returns:
            SymbolicScalarField: A new instance of SymbolicScalarField in the
            Cartesian coordinate system.

        Raises:
            ValueError: If the current coordinate system is not cylindrical or spherical.
        """
        if not isinstance(
            self.coord_system,
            (SymbolicCylindricalCoordSystem, SymbolicSphericalCoordSystem),
        ):
            raise ValueError(
                "Conversion to Cartesian is only implemented for cylindrical and spherical coordinate systems."
            )
        expr_dict = self.coord_system.get_basis_cartesian_exprs()
        cartesian_data = self.data.subs(expr_dict)
        cartesian_coord_system = SymbolicCartesianCoordSystem()
        return SymbolicScalarField(
            cartesian_data, cartesian_coord_system, self.field_params
        )

    def to_cylindrical(self):
        if not isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            raise NotImplementedError(
                "Conversion from non-Cartesian systems is not implemented"
            )
        expr_dict = self.coord_system.get_basis_cylindrical_exprs()
        cylindrical_data = self.data.subs(expr_dict)
        cylindrical_coord_system = SymbolicCylindricalCoordSystem()
        return self.__class__(
            cylindrical_data, cylindrical_coord_system, self.field_params
        )

    def to_spherical(self):
        if not isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            raise NotImplementedError(
                "Conversion from non-Cartesian systems is not implemented"
            )
        expr_dict = self.coord_system.get_basis_spherical_exprs()
        spherical_data = self.data.subs(expr_dict)
        spherical_coord_system = SymbolicSphericalCoordSystem()
        return self.__class__(spherical_data, spherical_coord_system, self.field_params)

    def subs(self, subs_dict, keys=False):
        try:
            if keys:
                for k, v in subs_dict.items():
                    if k in self.data:
                        self.data = self.data.subs({k: v})
                    else:
                        raise KeyError(f"Key '{k}' not found in data.")
            else:
                self.data = self.data.subs(subs_dict)
        except Exception as e:
            raise RuntimeError(f"An error occurred during substitution: {e}")


class SymbolicSpatialField(SymbolicField):
    def __init__(self, data, coord_system, field_params=None):
        if isinstance(data, sp.MutableDenseNDimArray):
            data = sp.ImmutableDenseNDimArray(data)

        if isinstance(data, (sp.Expr, sp.ImmutableDenseNDimArray)):
            super().__init__(data, coord_system, field_params)
        else:
            raise ValueError("Input data must be a SymPy Expr or SymPy Array")

    def lambdify(self):
        """
        Converts the symbolic field data into a lambda function for numerical evaluation.
        If the field is not in Cartesian coordinates, it first converts it to Cartesian.

        Returns:
            function: A lambda function for numerical evaluation of the field.
        """
        # Ensure the field is in Cartesian coordinates
        if not isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            field_in_cartesian = self.to_cartesian()
        else:
            field_in_cartesian = self

        basis_symbols = field_in_cartesian.coord_system.basis_symbols
        data = field_in_cartesian.data
        return sp.lambdify(basis_symbols, data, "numpy")


class SymbolicScalarField(SymbolicSpatialField):
    shape = (3,)

    @classmethod
    def create(cls, data=None, coord_system=None, field_params=None):
        if not data:
            if not coord_system:
                coord_system = SymbolicCartesianCoordSystem()
            data = sp.Function("f")(*coord_system.basis_symbols)
        else:
            if isinstance(data, sp.Expr):
                if not coord_system:
                    # autodetect the coord system
                    pass
                # validate the coord system

                # extract the params
            else:
                raise ValueError()

        return cls(data, coord_system, field_params)

    @classmethod
    def create_linear(cls, data, coord_system=None, field_params=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        if not isinstance(data, sp.ImmutableDenseNDimArray) or data.shape != cls.shape:
            raise ValueError("Data must be a 3 SymPy Array.")

        # Use the basis symbols from the coordinate system
        basis_symbols = coord_system.basis_symbols

        scalar_field = sum(var * coeff for coeff, var in zip(data, basis_symbols))
        return cls(scalar_field, coord_system, field_params)

    def plot(self, x_limits=[-100, 100], y_limits=[-100, 100], z_limits=[-100, 100]):
        if self.field_params:
            raise ValueError(
                "Cannot plot field with unresolved parameters: "
                + ", ".join(str(p) for p in self.field_params)
            )

        x_vals = np.linspace(*x_limits, 100)
        y_vals = np.linspace(*y_limits, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        f = self.lambdify()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Number of z slices
        num_slices = 10
        z_slices = np.linspace(z_limits[0], z_limits[1], num_slices)

        # Plot 2D contour plots at different z slices
        for z_val in z_slices:
            Z = f(X, Y, z_val)
            contour = ax.contourf(
                X, Y, Z, zdir="z", offset=z_val, levels=20, cmap="viridis", alpha=0.5
            )

        fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)

        ax.set_xlim(np.array(x_limits) * 1.2)
        ax.set_ylim(np.array(y_limits) * 1.2)
        ax.set_zlim(np.array(z_limits) * 1.2)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        plt.show()


class SymbolicVectorField(SymbolicSpatialField):
    shape = (3, 3)

    @classmethod
    def create(cls, data=None, coord_system=None, field_params=None):
        if not data:
            if not coord_system:
                coord_system = SymbolicCartesianCoordSystem()
            f1, f2, f3 = sp.symbols("f_1 f_2 f_3", cls=sp.Function)
            f1 = f1(*coord_system.basis_symbols)
            f2 = f2(*coord_system.basis_symbols)
            f3 = f3(*coord_system.basis_symbols)
            data = sp.ImmutableDenseNDimArray([f1, f2, f3])
        else:
            if not coord_system:
                coord_system = SymbolicCartesianCoordSystem()

            try:
                components = sp.ImmutableDenseNDimArray(data, shape=(3,))
                if not all(isinstance(_, (sp.Expr, sp.Number)) for _ in components):
                    raise ValueError("data type error")
            except:
                raise ValueError("Conversion error")
            
            data = sp.ImmutableDenseNDimArray(data)

        return cls(data, coord_system, field_params)

    @classmethod
    def create_linear(cls, data, coord_system=None, field_params=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        # Ensure data is a 3x3 SymPy Array
        if not isinstance(data, sp.ImmutableDenseNDimArray) or data.shape != cls.shape:
            raise ValueError("Data must be a 3x3 SymPy Array.")

        # Use the basis symbols from the coordinate system
        basis_symbols = coord_system.basis_symbols
        vector_field_components = [
            sum(data[i, j] * symbol for j, symbol in enumerate(basis_symbols))
            for i in range(3)
        ]

        vector_field = sp.ImmutableDenseNDimArray(vector_field_components)
        return cls(vector_field, coord_system, field_params)

    def plot(self, x_limits=[-100, 100], y_limits=[-100, 100], z_limits=[-100, 100]):
        # Create a grid of points within the specified limits
        x_vals = np.linspace(*x_limits, 10)
        y_vals = np.linspace(*y_limits, 10)
        z_vals = np.linspace(*z_limits, 10)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

        f = self.lambdify()

        # Evaluate the function at each point in the grid
        U, V, W = f(X, Y, Z)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the vector field using a quiver plot
        ax.quiver(X, Y, Z, U, V, W, length=0.01)

        # Set the limits and labels
        ax.set_xlim(np.array(x_limits) * 1.2)
        ax.set_ylim(np.array(y_limits) * 1.2)
        ax.set_zlim(np.array(z_limits) * 1.2)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        plt.show()


class SymbolicTensorField(SymbolicSpatialField):
    shape = (3, 3, 3)

    # @classmethod
    # def create(cls, coord_system=None, field_params=None):
    #     if not coord_system:
    #         coord_system = SymbolicCartesianCoordSystem()

    #     tensor_components = [
    #         [sp.Function(f"f_{i}{j}")(*coord_system.basis_symbols) for j in range(3)]
    #         for i in range(3)
    #     ]
    #     tensor_field = sp.tensor.Array(tensor_components)
    #     return cls(tensor_field, coord_system, field_params)

    @classmethod
    def create_linear(cls, data, coord_system=None, field_params=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        # Ensure data is a 3x3x3 SymPy Array
        if not isinstance(data, sp.ImmutableDenseNDimArray) or data.shape != cls.shape:
            raise ValueError("Data must be a 3x3x3 SymPy Array.")

        # Use the basis symbols from the coordinate system
        basis_symbols = coord_system.basis_symbols

        # Construct the tensor field components as linear combinations of basis symbols
        tensor_field_components = []
        for i in range(3):  # For the first index
            tensor_row = []
            for j in range(3):  # For the second index
                component_expr = sum(
                    data[i, j, k] * symbol for k, symbol in enumerate(basis_symbols)
                )
                tensor_row.append(component_expr)
            tensor_field_components.append(tensor_row)

        tensor_field = sp.ImmutableDenseNDimArray(tensor_field_components)
        return cls(tensor_field, coord_system, field_params)
