import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp


class SymbolicField:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.data}\n)"


class SymbolicField3D(SymbolicField):
    def __init__(self, data):
        if isinstance(data, sp.MutableDenseNDimArray):
            data = sp.ImmutableDenseNDimArray(data)
        if isinstance(data, (sp.Expr, sp.ImmutableDenseNDimArray)):
            self.validate_spatial_variables(data)
            super().__init__(data)
        else:
            raise ValueError("Input data must be a SymPy Expr or SymPy Array")

    def validate_spatial_variables(self, data):
        free_symbols = (
            data.free_symbols
            if isinstance(data, sp.Expr)
            else set().union(*[element.free_symbols for element in data])
        )
        if not all(var in free_symbols for var in sp.symbols("x y z")):
            raise ValueError(
                "Symbolic data must be a function of spatial variables x, y, z"
            )


class SymbolicScalarField(SymbolicField3D):
    shape = (3,)

    @classmethod
    def create(cls):
        x, y, z = sp.symbols("x y z")
        f = sp.symbols("f", cls=sp.Function)
        f = f(x, y, z)
        return cls(f)

    @classmethod
    def create_linear(cls, data):
        if not isinstance(data, sp.ImmutableDenseNDimArray) or data.shape != cls.shape:
            raise ValueError("Data must be a 3 SymPy Array.")

        x, y, z = sp.symbols("x y z")
        scalar_field = sum(var * coeff for coeff, var in zip(data, [x, y, z]))
        return cls(scalar_field)

    def plot(self, x_limits=[-100, 100], y_limits=[-100, 100], z_limits=[-100, 100]):
        x_vals = np.linspace(*x_limits, 100)
        y_vals = np.linspace(*y_limits, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        f = sp.lambdify(sp.symbols("x y z"), self.data, "numpy")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Number of z slices
        num_slices = 10
        z_slices = np.linspace(z_limits[0], z_limits[1], num_slices)

        # Plot 2D contour plots at different z slices
        for z_val in z_slices:
            Z = f(X, Y, z_val)
            ax.contourf(
                X, Y, Z, zdir="z", offset=z_val, levels=20, cmap="viridis", alpha=0.5
            )

        ax.set_xlim(np.array(x_limits) * 1.2)
        ax.set_ylim(np.array(y_limits) * 1.2)
        ax.set_zlim(np.array(z_limits) * 1.2)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        plt.show()


class SymbolicVectorField(SymbolicField3D):
    @classmethod
    def create(cls):
        x, y, z = sp.symbols("x y z")
        fx, fy, fz = sp.symbols("f_x f_y f_z", cls=sp.Function)
        fx = fx(x, y, z)
        fy = fy(x, y, z)
        fz = fz(x, y, z)
        vector_field = sp.Array([fx, fy, fz])
        return cls(vector_field)

    @classmethod
    def create_linear(cls, data):
        if not isinstance(data, sp.ImmutableDenseNDimArray) or data.shape != (3, 3):
            raise ValueError("Data must be a 3x3 SymPy Array.")

        x, y, z = sp.symbols("x y z")
        fx = sum(data[0, i] * var for i, var in enumerate([x, y, z]))
        fy = sum(data[1, i] * var for i, var in enumerate([x, y, z]))
        fz = sum(data[2, i] * var for i, var in enumerate([x, y, z]))

        vector_field = sp.Array([fx, fy, fz])
        return cls(vector_field)

    def plot(self, x_limits=[-100, 100], y_limits=[-100, 100], z_limits=[-100, 100]):
        # Create a grid of points within the specified limits
        x_vals = np.linspace(*x_limits, 10)
        y_vals = np.linspace(*y_limits, 10)
        z_vals = np.linspace(*z_limits, 10)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

        # Convert the symbolic expressions to numerical functions
        f = sp.lambdify(sp.symbols("x y z"), self.data, "numpy")

        # Evaluate the function at each point in the grid
        U, V, W = f(X, Y, Z)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Plot the vector field using a quiver plot
        ax.quiver(X, Y, Z, U, V, W, length=0.01)

        # Set the limits and labels
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")

        plt.show()


class SymbolicTensorField(SymbolicField3D):
    @classmethod
    def create(cls):
        x, y, z = sp.symbols("x y z")
        tensor_components = [
            [sp.Function(f"f_{i}{j}")(x, y, z) for j in range(3)] for i in range(3)
        ]
        tensor_field = sp.Array(tensor_components)
        return cls(tensor_field)

    @classmethod
    def create_linear(cls, data):
        if not isinstance(data, sp.ImmutableDenseNDimArray) or data.shape != (3, 3, 3):
            raise ValueError("Data must be a 3x3x3 SymPy Array.")

        x, y, z = sp.symbols("x y z")
        tensor_field = sp.Array(
            [
                [
                    sum(data[i, j, k] * var for k, var in enumerate([x, y, z]))
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )

        return cls(tensor_field)
