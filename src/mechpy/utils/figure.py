import copy
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RangeSlider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

from mechpy.core.symbolic.field import (
    SymbolicField,
    SymbolicScalarField,
    SymbolicVectorField,
)


class Figure:
    def __init__(self, fig_type=None):
        if not fig_type:
            fig_type = "2D"

        self.fig = plt.figure(figsize=(6, 8))
        if fig_type == "2D":
            self.ax = self.fig.add_subplot(211)
        elif fig_type == "3D":
            self.ax = self.fig.add_subplot(211, projection="3d")
        else:
            raise ValueError(f"figure type: {fig_type} not implemented")


    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SymbolicScalarFieldFigure(SymbolicScalarField, Figure):
    def __init__(self, field):
        if not isinstance(field, SymbolicScalarField):
            field_class_name = self.__class__.__bases__[0].__name__
            raise TypeError(f"field should be {field_class_name} not, {type(field)}")
        self.field = field
        Figure.__init__(self, fig_type="3D")

    def plot(self, limits=None):
        if limits == None:
            x_limits = [-20, 20]
            y_limits = [-20, 20]
            z_limits = [-20, 20]

        x_vals = np.linspace(*x_limits, 20)
        y_vals = np.linspace(*y_limits, 20)
        X, Y = np.meshgrid(x_vals, y_vals)

        self.ax.set_xlim(np.array(x_limits) * 1.2)
        self.ax.set_ylim(np.array(y_limits) * 1.2)
        self.ax.set_zlim(np.array(z_limits) * 1.2)
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.set_zlabel("Z axis")

        self.sliders = {}
        for i, (k, v) in enumerate(self.field.field_params.items()):
            ax = plt.axes([0.15, 0.4 - 0.05 * i, 0.7, 0.02])
            slider = Slider(
                ax, str(k), min(v), max(v), valstep=v, valinit=v[(len(v) + 1) // 2]
            )
            self.sliders.update({k: slider})

        def update(val):
            if hasattr(self, "colorbar"):
                self.colorbar.remove()
                del self.colorbar

            self.ax.clear()
            self.ax.set_zlim(np.array(z_limits) * 1.2)

            field = copy.deepcopy(self.field)
            subs_dict = {param: slider.val for param, slider in self.sliders.items()}
            field.subs(subs_dict)
            f = field.lambdify()

            num_slices = 5
            z_slices = np.linspace(z_limits[0], z_limits[1], num_slices)

            for z_val in z_slices:
                contour = self.ax.contourf(
                    X,
                    Y,
                    f(X, Y, z_val),
                    zdir="z",
                    offset=z_val,
                    levels=20,
                    alpha=0.5,
                    vmin=-100,
                    vmax=100,
                )

            self.colorbar = self.fig.colorbar(contour, ax=self.ax, shrink=0.5, aspect=5)

            plt.draw()

        update(None)

        if hasattr(self, "sliders"):
            for slider in self.sliders.values():
                slider.on_changed(update)

        self.fig.show()


class SymbolicVectorFieldFigure(SymbolicVectorField, Figure):
    def __init__(self, field):
        if not isinstance(field, SymbolicVectorField):
            field_class_name = self.__class__.__bases__[0].__name__
            raise TypeError(f"field should be {field_class_name} not, {type(field)}")
        self.field = field
        Figure.__init__(self, fig_type="3D")

    def plot(self, limits=None):
        if limits == None:
            x_limits = [-20, 20]
            y_limits = [-20, 20]
            z_limits = [-20, 20]

        x_vals = np.linspace(*x_limits, 10)
        y_vals = np.linspace(*y_limits, 10)
        z_vals = np.linspace(*z_limits, 10)
        X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals)

        self.ax.set_xlim(np.array(x_limits) * 1.2)
        self.ax.set_ylim(np.array(y_limits) * 1.2)
        self.ax.set_zlim(np.array(z_limits) * 1.2)
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.set_zlabel("Z axis")

        self.sliders = {}
        for i, (k, v) in enumerate(self.field.field_params.items()):
            ax = plt.axes([0, 0 + 0.05 * i, 0.5, 0.1])
            slider = Slider(
                ax, str(k), min(v), max(v), valstep=v, valinit=v[(len(v) + 1) // 2]
            )
            self.sliders.update({k: slider})

        def update(val):
            if hasattr(self, "colorbar"):
                self.colorbar.remove()
                del self.colorbar

            self.ax.clear()

            field = copy.deepcopy(self.field)
            subs_dict = {param: slider.val for param, slider in self.sliders.items()}
            field.subs(subs_dict)
            f = field.lambdify()
            U, V, W = f(X, Y, Z)

            # Plot the vector field using a quiver plot
            self.ax.quiver(X, Y, Z, U, V, W, length=0.01)

            plt.draw()

        update(None)

        if hasattr(self, "sliders"):
            for slider in self.sliders.values():
                slider.on_changed(update)

        self.fig.show()
