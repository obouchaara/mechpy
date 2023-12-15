import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp


from .coord import (
    SymbolicCoordSystem,
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)


class SymbolicField:
    def __init__(self, data, coord_system):
        if isinstance(coord_system, SymbolicCoordSystem):
            self.coord_system = coord_system
            self.data = data
        else:
            raise ValueError("coord system must be a SymbolicCoordSystem")

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data},\n{self.coord_system.basis_symbols}\n)"

    def to_cylindrical(self):
        if isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            expr_dict = self.coord_system.get_basis_cylindrical_exprs()
        else:
            raise NotImplementedError(
                "Conversion from non-Cartesian systems is not implemented"
            )

        cylindrical_data = self.data.subs(expr_dict)
        cylindrical_coord_system = SymbolicCylindricalCoordSystem()
        return self.__class__(cylindrical_data, cylindrical_coord_system)


class SymbolicField3D(SymbolicField):
    def __init__(self, data, coord_system):
        if isinstance(data, sp.MutableDenseNDimArray):
            data = sp.ImmutableDenseNDimArray(data)

        if isinstance(data, (sp.Expr, sp.ImmutableDenseNDimArray)):
            super().__init__(data, coord_system)
        else:
            raise ValueError("Input data must be a SymPy Expr or SymPy Array")


class SymbolicScalarField(SymbolicField3D):
    shape = (3,)

    @classmethod
    def create(cls, coord_system=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        f = sp.Function("f")(*coord_system.basis_symbols)
        return cls(f, coord_system)

    @classmethod
    def create_linear(cls, data, coord_system=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        if not isinstance(data, sp.ImmutableDenseNDimArray) or data.shape != cls.shape:
            raise ValueError("Data must be a 3 SymPy Array.")

        # Use the basis symbols from the coordinate system
        basis_symbols = coord_system.basis_symbols

        scalar_field = sum(var * coeff for coeff, var in zip(data, basis_symbols))
        return cls(scalar_field, coord_system)

    def plot(self, x_limits=[-100, 100], y_limits=[-100, 100], z_limits=[-100, 100]):
        x, y, z = sp.symbols("x y z")

        x_vals = np.linspace(*x_limits, 100)
        y_vals = np.linspace(*y_limits, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        if isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            data = self.data
        elif isinstance(
            self.coord_system,
            (SymbolicCylindricalCoordSystem, SymbolicSphericalCoordSystem),
        ):
            expr_dict = self.coord_system.get_basis_cartesian_exprs()
            data = self.data.subs(expr_dict)
        else:
            raise ValueError("Unsupported coordinate system. The coordinate system.")

        f = sp.lambdify((x, y, z), data, "numpy")

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


class SymbolicVectorField(SymbolicField3D):
    shape = (3, 3)

    @classmethod
    def create(cls, coord_system=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        f1, f2, f3 = sp.symbols("f_1 f_2 f_3", cls=sp.Function)
        f1 = f1(*coord_system.basis_symbols)
        f2 = f2(*coord_system.basis_symbols)
        f3 = f3(*coord_system.basis_symbols)
        vector_field = sp.ImmutableDenseNDimArray([f1, f2, f3])
        return cls(vector_field, coord_system)

    @classmethod
    def create_linear(cls, data, coord_system=None):
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
        return cls(vector_field, coord_system)

    def plot(self, x_limits=[-100, 100], y_limits=[-100, 100], z_limits=[-100, 100]):
        # Create a grid of points within the specified limits
        x_vals = np.linspace(*x_limits, 10)
        y_vals = np.linspace(*y_limits, 10)
        z_vals = np.linspace(*z_limits, 10)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

        if isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            data = self.data
        elif isinstance(
            self.coord_system,
            (SymbolicCylindricalCoordSystem, SymbolicSphericalCoordSystem),
        ):
            expr_dict = self.coord_system.get_basis_cartesian_exprs()
            data = self.data.subs(expr_dict)
        else:
            raise ValueError("Unsupported coordinate system. The coordinate system.")

        # Convert the symbolic expressions to numerical functions
        f = sp.lambdify(sp.symbols("x y z"), data, "numpy")

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


class SymbolicTensorField(SymbolicField3D):
    shape = (3, 3, 3)

    @classmethod
    def create(cls, coord_system=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        tensor_components = [
            [sp.Function(f"f_{i}{j}")(*coord_system.basis_symbols) for j in range(3)]
            for i in range(3)
        ]
        tensor_field = sp.tensor.Array(tensor_components)
        return cls(tensor_field, coord_system)

    @classmethod
    def create_linear(cls, data, coord_system=None):
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
        return cls(tensor_field, coord_system)
