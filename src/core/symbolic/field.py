import sympy as sp


class SymbolicField:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"SymbolicField(\n{self.data}\n)"


class SymbolicField3D(SymbolicField):
    def __init__(self, data):
        if isinstance(data, (sp.Expr, sp.Array)):
            free_symbols = data.free_symbols if isinstance(data, sp.Expr) else set().union(*[element.free_symbols for element in data])
            if all(var in free_symbols for var in sp.symbols("x y z")):
                super().__init__(data)
            else:
                raise ValueError("Symbolic data must be a function of spatial variables x, y, z")
        else:
            raise ValueError("Input data must be a SymPy Expr or SymPy Array")

    def __repr__(self):
        return f"SymbolicField3D(\n{self.data}\n)"


class SymbolicScalarField(SymbolicField3D):
    def __repr__(self):
        return f"SymbolicScalarField(\n{self.data}\n)"

    @classmethod
    def create(cls):
        x, y, z = sp.symbols("x y z")
        f = sp.symbols("f", cls=sp.Function)
        f = f(x, y, z)
        return cls(f)

    @classmethod
    def create_linear(cls, data):
        if not isinstance(data, sp.Array) or data.shape != (3,):
            raise ValueError("Data must be a 3 SymPy Array.")

        x, y, z = sp.symbols("x y z")
        scalar_field = sum(data[i] * var for i, var in enumerate([x, y, z]))
        return cls(scalar_field)


class SymbolicVectorField(SymbolicField3D):
    def __repr__(self):
        return f"SymbolicVectorField(\n{self.data}\n)"

    @classmethod
    def create(cls):
        x, y, z = sp.symbols("x y z")
        fx, fy, fz = sp.symbols("f_x f_y f_z", cls=sp.Function)
        fx = fx(x, y, z)
        fy = fy(x, y, z)
        fz = fz(x, y, z)
        verctor_field = sp.Array([fx, fy, fz])
        return cls(verctor_field)

    @classmethod
    def create_linear(cls, data):
        if not isinstance(data, sp.Array) or data.shape != (3, 3):
            raise ValueError("Data must be a 3x3 SymPy Array.")

        x, y, z = sp.symbols("x y z")
        fx = sum(data[0, i] * var for i, var in enumerate([x, y, z]))
        fy = sum(data[1, i] * var for i, var in enumerate([x, y, z]))
        fz = sum(data[2, i] * var for i, var in enumerate([x, y, z]))

        verctor_field = sp.Array([fx, fy, fz])
        return cls(verctor_field)


class SymbolicTensorField(SymbolicField3D):
    def __repr__(self):
        return f"SymbolicTensorField(\n{self.data}\n)"

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
        if not isinstance(data, sp.Array) or data.shape != (3, 3, 3):
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
