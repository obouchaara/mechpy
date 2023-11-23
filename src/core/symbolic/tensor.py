import sympy as sp


class SymblocTensor:
    pass


class SymbolicThreeByThreeTensor(SymblocTensor):
    def __init__(self, data):
        if not isinstance(data, sp.Matrix) or data.shape != (3, 3):
            raise ValueError("Input data must be a 3x3 SymPy Matrix")

        self.data = data

    def __repr__(self):
        return f"SymbolicThreeByThreeTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        if isinstance(components, list):
            if len(components) == 9:
                data = sp.Matrix(components).reshape(3, 3)
                return cls(data)
            elif len(components) == 3 and all(
                isinstance(row, list) and len(row) == 3 for row in components
            ):
                data = sp.Matrix(components)
                return cls(data)
            else:
                raise ValueError("Invalid input for creating a 3x3 tensor")
        else:
            raise ValueError("Input must be a list")

    def is_symmetric(self):
        return self.data == self.data.transpose()

    def to_symmetric(self):
        if self.is_symmetric():
            data = sp.Matrix(
                [
                    self.data[0, 0],
                    self.data[1, 1],
                    self.data[2, 2],
                    self.data[2, 1],
                    self.data[2, 0],
                    self.data[1, 0],
                ]
            )
            return SymbolicSymmetricThreeByThreeTensor(data)
        raise ValueError(
            "The tensor is not symmetric, and it cannot be converted to a symmetric tensor."
        )


class SymbolicSixBySixTensor(SymblocTensor):
    pass

class SymbolicSymmetricThreeByThreeTensor:
    def __init__(self, data):
        if not isinstance(data, sp.Matrix) or data.shape != (6, 1):
            raise ValueError("Input data must be a 6x1 SymPy Matrix")

        self.data = data

    def __repr__(self):
        return f"SymmetricSymbolicThreeByThreeTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        if isinstance(components, list) and len(components) == 6:
            data = sp.Matrix(components)
            return cls(data)
        else:
            raise ValueError(
                "Invalid input for creating a symbolic symmetric 3x3 tensor. Provide a list of 6 symbolic components."
            )

    def to_general_tensor(self):
        components = [
            [self.data[0], self.data[5], self.data[4]],
            [self.data[5], self.data[1], self.data[3]],
            [self.data[4], self.data[3], self.data[2]],
        ]
        data = sp.Matrix(components)
        return SymbolicThreeByThreeTensor(data)
