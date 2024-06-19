import copy
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from .coord import (
    SymbolicCoordSystem,
    SymbolicCartesianCoordSystem,
    SymbolicCylindricalCoordSystem,
    SymbolicSphericalCoordSystem,
)


class SymbolicField:
    """
    Represents a symbolic field in a given coordinate system.

    :param coord_system: The coordinate system for the field.
    :type coord_system: SymbolicCoordSystem
    :param data: The field data.
    :type data: sp.NDimArray
    :param field_params: Additional field parameters. Defaults to {}.
    :type field_params: dict, optional
    :param symbols_validation: If True, validate symbols in the field data. Defaults to True.
    :type symbols_validation: bool

    :raises ValueError: If coord_system is not a SymbolicCoordSystem, data is not a NDimArray,
                        or if field parameters overlap with coordinate system basis symbols.
    """

    def __init__(self, coord_system, data, field_params=None, symbols_validation=True):
        if not isinstance(coord_system, SymbolicCoordSystem):
            raise ValueError("Coord system must be a SymbolicCoordSystem.")

        if not isinstance(data, sp.NDimArray):
            raise ValueError("Data must be a NDimArray.")

        self.coord_system = coord_system
        self.data = data
        self.field_params = field_params or {}

        if not isinstance(self.field_params, dict):
            raise ValueError("Field parameters must be a dict.")

        basis = set(self.coord_system.basis)
        field_param = set(self.field_params)
        if not field_param.isdisjoint(basis):
            raise ValueError(
                "Field parameters must not overlap with coordinate system basis symbols."
            )

        if symbols_validation:
            invalid_symbols = self.get_invalid_symbols()
            if len(invalid_symbols):
                raise ValueError(
                    "The field data contains symbols not in the basis or field parameters: "
                    + ", ".join(str(s) for s in invalid_symbols)
                )

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self.coord_system.basis},\n{self.data},\n{self.field_params})"

    def get_invalid_symbols(self) -> set:
        """
        Get symbols in the field data that are not in the basis or field parameters.

        :return: A set of invalid symbols.
        :rtype: set
        """

        def get_ignored_symbols(data):
            symbols_set = set()
            # to implement
            return symbols_set

        basis = set(self.coord_system.basis)
        field_param = set(self.field_params)
        valid_symbols = basis.union(field_param)
        free_symbols = self.data.free_symbols
        free_symbols = {s for s in free_symbols if not isinstance(s, sp.Number)}
        ignored_symbols = get_ignored_symbols(self.data)
        invalid_symbols = free_symbols - valid_symbols - ignored_symbols
        return invalid_symbols

    def subs_field_params(self, param_values):
        """
        Substitute the provided field parameters with specific values, and
        remove them from self.field_params.

        :param param_values: A dictionary mapping parameters to their values.
        :type param_values: dict
        :raises TypeError: If param_values is not a dictionary.
        :raises ValueError: If a parameter in param_values is not in self.field_params.

        :return: None. The method updates self.data and self.field_params in place.
        """
        if not isinstance(param_values, dict):
            raise TypeError("param_values must be a dictionary")

        for param, value in param_values.items():
            if not param in self.field_params:
                raise ValueError(f"Parameter {param} not found in field parameters")

            self.data = self.data.subs(param, value)
            self.field_params.pop(param)

    def to_cartesian(self):
        """
        Converts the scalar field from its current coordinate system
        (cylindrical or spherical) to the Cartesian coordinate system.

        :raises ValueError: If the current coordinate system is not cylindrical or spherical.

        :return: A new SymbolicField in the Cartesian coordinate system.
        :rtype: SymbolicField
        """
        coord_classes = (SymbolicCylindricalCoordSystem, SymbolicSphericalCoordSystem)
        if not isinstance(self.coord_system, coord_classes):
            raise ValueError(
                "Conversion to Cartesian is only implemented for cylindrical and spherical coordinate systems."
            )
        cartesian_coord_system = SymbolicCartesianCoordSystem()
        expr_dict = cartesian_coord_system.get_basis_cylindrical_exprs()
        cartesian_data = self.data.subs(expr_dict)
        return SymbolicScalarField(
            cartesian_coord_system, cartesian_data, self.field_params
        )

    def to_cylindrical(self):
        """
        Converts the scalar field from its current coordinate system
        (Cartesian) to the cylindrical coordinate system.

        :raises NotImplementedError: If the current coordinate system is not Cartesian.

        :return: A new SymbolicField in the cylindrical coordinate system.
        :rtype: SymbolicField
        """
        if not isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            raise NotImplementedError(
                "Conversion from non-Cartesian systems is not implemented"
            )
        cylindrical_coord_system = SymbolicCylindricalCoordSystem()
        expr_dict = cylindrical_coord_system.get_basis_cartesian_exprs()
        cylindrical_data = self.data.subs(expr_dict)
        return self.__class__(
            cylindrical_coord_system, cylindrical_data, self.field_params
        )

    def to_spherical(self):
        """
        Converts the scalar field from its current coordinate system
        (Cartesian) to the spherical coordinate system.

        :raises NotImplementedError: If the current coordinate system is not Cartesian.

        :return: A new SymbolicField in the spherical coordinate system.
        :rtype: SymbolicField
        """
        if not isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            raise NotImplementedError(
                "Conversion from non-Cartesian systems is not implemented"
            )
        spherical_coord_system = SymbolicSphericalCoordSystem()
        expr_dict = spherical_coord_system.get_basis_cartesian_exprs()
        spherical_data = self.data.subs(expr_dict)
        return self.__class__(spherical_coord_system, spherical_data, self.field_params)

    def subs(self, subs_dict, keys=False):
        """
        Deprecated method for substituting values. Raises a warning.

        :raises DeprecationWarning: This method is deprecated.
        """
        raise DeprecationWarning


class SymbolicSpatialField(SymbolicField):
    def __init__(self, coord_system, data, field_params=None, symbols_validation=True):
        super().__init__(coord_system, data, field_params, symbols_validation)

    def lambdify(self):
        """
        Converts the symbolic field data into a lambda function for numerical evaluation.
        If the field is not in Cartesian coordinates, it first converts it to Cartesian.
        """
        # Ensure the field is in Cartesian coordinates
        if not isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            field_in_cartesian = self.to_cartesian()
        else:
            field_in_cartesian = self

        basis = field_in_cartesian.coord_system.basis
        data = field_in_cartesian.data
        return sp.lambdify(basis, data, "numpy")


class SymbolicScalarField(SymbolicSpatialField):
    shape = (1,)

    @classmethod
    def create(cls, coord_system=None, data=None, field_params=None):
        if not data:
            if not coord_system:
                coord_system = SymbolicCartesianCoordSystem()
            data = sp.NDimArray([sp.Function("f")(*coord_system.basis)])

        if isinstance(data, sp.NDimArray):
            return cls(coord_system, data, field_params)
        else:
            raise NotImplementedError
            # if not coord_system:
            #     # autodetect the coord system
            #     pass
            # # validate the coord system

            # # extract the params
            # else:
            #     raise ValueError()

    @classmethod
    def create_linear(cls, coord_system=None, data=None, field_params=None):
        if coord_system is None:
            coord_system = SymbolicCartesianCoordSystem()

        if not isinstance(data, sp.NDimArray) or data.shape != (3,):
            raise ValueError("Data must be a 3 SymPy Array.")

        # Use the basis symbols from the coordinate system
        basis = coord_system.basis
        scalar_field = sp.NDimArray(
            [data[0] * basis[0] + data[1] * basis[1] + data[2] * basis[2]]
        )
        return cls(coord_system, scalar_field, field_params)

    def plot(
        self, x_range=(-10, 10), y_range=(-10, 10), z_range=(-10, 10), num_points=20
    ):
        if not isinstance(self.coord_system, SymbolicCartesianCoordSystem):
            raise NotImplementedError(
                "Plotting is only implemented for Cartesian coordinates"
            )

        # Create a meshgrid for the plot
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        z = np.linspace(z_range[0], z_range[1], num_points)
        X, Y, Z = np.meshgrid(x, y, z)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(
            left=0.1, bottom=0.25 + 0.05 * len(self.field_params)
        )  # Adjust space for sliders

        sliders = {}
        slider_axes = []

        # Create sliders for each field parameter
        for i, (param, values) in enumerate(self.field_params.items()):
            if values is None:
                raise ValueError(f"the param {param} values in not defined")
            if isinstance(values, set):
                ax_slider = plt.axes(
                    [0.1, 0.1 + 0.05 * i, 0.65, 0.03], facecolor="lightgoldenrodyellow"
                )
                slider = Slider(
                    ax_slider,
                    str(param),
                    min(values),
                    max(values),
                    valinit=min(values),
                    valstep=list(values),
                )
                sliders[param] = slider
                slider_axes.append(ax_slider)
            else:
                raise NotImplementedError

        # Initial parameter values
        param_values = {param: slider.val for param, slider in sliders.items()}
        field = copy.deepcopy(self)
        field.subs_field_params(param_values)
        f = field.lambdify()

        values = f(X, Y, Z)
        scatter = ax.scatter(X, Y, Z, c=values, cmap="viridis")
        colorbar = fig.colorbar(scatter, ax=ax, label="Field Value")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Scalar Field")

        def update(val):
            param_values = {param: slider.val for param, slider in sliders.items()}
            field = copy.deepcopy(self)
            field.subs_field_params(param_values)
            f = field.lambdify()

            values = f(X, Y, Z)
            scatter.set_array(values)
            scatter.set_offsets(np.c_[X.flatten(), Y.flatten(), Z.flatten()])
            scatter.set_array(values.flatten())
            scatter.changed()
            fig.canvas.draw_idle()

        for slider in sliders.values():
            slider.on_changed(update)

        # Initial plot
        update(None)
        plt.show()


class SymbolicVectorField(SymbolicSpatialField):
    shape = (3,)

    @classmethod
    def create(
        cls,
        coord_system=None,
        data=None,
        field_params=None,
        symbols_validation=True,
    ):
        if data is None:
            if coord_system is None:
                coord_system = SymbolicCartesianCoordSystem()
            f1, f2, f3 = sp.symbols("f_1 f_2 f_3", cls=sp.Function)
            basis = coord_system.basis
            f1 = f1(*basis)
            f2 = f2(*basis)
            f3 = f3(*basis)
            data = sp.NDimArray([f1, f2, f3])
        else:
            if coord_system is None:
                coord_system = SymbolicCartesianCoordSystem()
            try:
                components = sp.NDimArray(data, shape=(3,))
                is_symbolic = lambda _: isinstance(
                    _, (sp.Number, sp.Symbol, sp.Expr)
                )  # to validation module
                if not all(is_symbolic(_) for _ in components):
                    raise ValueError("data type error")
            except:
                raise ValueError("Conversion error")

            data = sp.NDimArray(data, shape=(3,))

        return cls(coord_system, data, field_params, symbols_validation)

    @classmethod
    def create_linear(cls, coord_system=None, data=None, field_params=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        if not isinstance(data, sp.NDimArray) or data.shape != (3, 3):
            raise ValueError("Data must be a 3x3 SymPy Array.")

        # Use the basis symbols from the coordinate system
        basis = coord_system.basis
        vector_field_components = [
            sum(data[i, j] * symbol for j, symbol in enumerate(basis)) for i in range(3)
        ]

        vector_field = sp.NDimArray(vector_field_components)
        return cls(coord_system, vector_field, field_params)

    def plot(self):
        pass


class SymbolicTensorField(SymbolicSpatialField):
    shape = (3, 3)

    # @classmethod
    # def create(cls, coord_system=None, field_params=None):
    #     if not coord_system:
    #         coord_system = SymbolicCartesianCoordSystem()

    #     tensor_components = [
    #         [sp.Function(f"f_{i}{j}")(*coord_system.basis) for j in range(3)]
    #         for i in range(3)
    #     ]
    #     tensor_field = sp.tensor.Array(tensor_components)
    #     return cls(tensor_field, coord_system, field_params)

    @classmethod
    def create_linear(cls, coord_system=None, data=None, field_params=None):
        if not coord_system:
            coord_system = SymbolicCartesianCoordSystem()

        if not isinstance(data, sp.NDimArray) or data.shape != (3, 3, 3):
            raise ValueError("Data must be a 3x3x3 SymPy Array.")

        basis = coord_system.basis

        # Construct the tensor field components as linear combinations of basis symbols
        tensor_field_components = []
        for i in range(3):  # For the first index
            tensor_row = []
            for j in range(3):  # For the second index
                component_expr = sum(
                    data[i, j, k] * symbol for k, symbol in enumerate(basis)
                )
                tensor_row.append(component_expr)
            tensor_field_components.append(tensor_row)

        tensor_field = sp.NDimArray(tensor_field_components)
        return cls(coord_system, tensor_field, field_params)
