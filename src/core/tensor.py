import math
import numpy as np
import sympy as sp


class Tensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError("Input data must be a NumPy array")

    def is_symmetric(self):
        return np.array_equal(self.data, self.data.T)

    def is_second_rank(self):
        return self.data.ndim == 2

    def is_fourth_rank(self):
        return self.data.ndim == 4

    def is_3x3(self):
        return self.data.shape == (3, 3)

    def is_6x6(self):
        return self.data.shape == (6, 6)

    def is_3x3x3x3(self):
        return self.data.shape == (3, 3, 3, 3)

    @classmethod
    def from_list(cls, components, shape, is_symmetric=False, symmetric_mode=1):
        if isinstance(components, list):
            if is_symmetric:
                if len(shape) == 2 and symmetric_mode == 1:
                    if len(components) != shape[0] * (shape[1] + 1) // 2:
                        raise ValueError("Invalid components for symmetric tensor")

                    data = np.zeros(shape)
                    index = 0
                    for i in range(shape[0]):
                        for j in range(i, shape[1]):
                            data[i, j] = data[j, i] = components[index]
                            index += 1
                    return cls(data)
                else:
                    raise NotImplementedError("Not implemented")
            if isinstance(shape, tuple):
                try:
                    data = np.array(components).reshape(shape)
                except:
                    raise ValueError("Invalid components or shape value")
                return cls(data)

            else:
                raise ValueError("Invalid shape parameter")
        else:
            raise ValueError("Input must be a list")


class ThreeByThreeTensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray) and data.shape == (3, 3):
            self.data = data
        else:
            raise ValueError("Input data must be a 3x3 NumPy array")

    def __repr__(self):
        return f"ThreeByThreeTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        if isinstance(components, list):
            if len(components) == 9:
                # Create a 3x3 tensor from a flat list of 9 components
                data = np.array(components).reshape((3, 3))
                return cls(data)
            elif len(components) == 3 and all(
                isinstance(row, list) and len(row) == 3 for row in components
            ):
                # Create a 3x3 tensor from a list of 3 lists
                data = np.array(components)
                return cls(data)
            else:
                raise ValueError("Invalid input for creating a 3x3 tensor")
        else:
            raise ValueError("Input must be a list")

    def is_symmetric(self):
        return np.array_equal(self.data, self.data.T)

    def to_symmetric(self):
        if self.is_symmetric():
            data = np.array(
                [
                    self.data[0, 0],
                    self.data[1, 1],
                    self.data[2, 2],
                    self.data[2, 1],
                    self.data[2, 0],
                    self.data[1, 0],
                ]
            )
            data = data.reshape((6, 1))  # Ensure the shape is (6, 1)
            return SymmetricThreeByThreeTensor(data)
        raise ValueError(
            "The tensor is not symmetric, and it cannot be converted to a symmetric tensor."
        )


class SixBySixTensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray) and data.shape == (6, 6):
            self.data = data
        else:
            raise ValueError("Input data must be a 6x6 NumPy array")

    def __repr__(self):
        return f"SixBySixTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        if isinstance(components, list):
            if len(components) == 36:
                data = np.array(components).reshape((6, 6))
                return cls(data)
            elif len(components) == 6 and all(
                isinstance(row, list) and len(row) == 6 for row in components
            ):
                data = np.array(components)
                return cls(data)
            else:
                raise ValueError("Invalid input for creating a 6x6 tensor")
        else:
            raise ValueError("Input must be a list")

    def is_symmetric(self):
        return np.array_equal(self.data, self.data.T)

    def to_symmetric(self):
        pass


class SymmetricThreeByThreeTensor:
    def __init__(self, data):
        if isinstance(data, np.ndarray) and data.shape in [(6,), (6, 1)]:
            self.data = data
        else:
            raise ValueError("Input data must be a 6x1 tensor")

    def __repr__(self):
        return f"SymmetricThreeByThreeTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        if isinstance(components, list) and len(components) == 6:
            data = np.array(components)
            return cls(data)
        else:
            raise ValueError(
                "Invalid input for creating a symmetric 3x3 tensor. Provide a list of 6 components."
            )

    def to_general_tensor(self):
        components = [
            [self.data[0], self.data[5], self.data[4]],
            [self.data[5], self.data[1], self.data[3]],
            [self.data[4], self.data[3], self.data[2]],
        ]
        data = np.array(components)
        data = data.reshape((3, 3))
        return ThreeByThreeTensor(data)


class SymbolicThreeByThreeTensor:
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
