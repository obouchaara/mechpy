import numpy as np


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
    def from_list(cls, components, shape, symmetric=False, mode=1):
        if isinstance(components, list):
            if symmetric:
                if shape == (3, 3) and mode == 1:
                    if len(components) != shape[0] * (shape[1] + 1) // 2:
                        raise ValueError("Invalid components for symmetric tensor")
                    components = [float(_) for _ in components]
                    data = np.array(
                        [
                            [components[0], components[5], components[4]],
                            [components[5], components[1], components[3]],
                            [components[4], components[3], components[2]],
                        ]
                    )
                    return cls(data)
                elif len(shape) == 2 and mode == 2:
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


class ThreeByThreeTensor(Tensor):
    shape = (3, 3)

    def __init__(self, data):
        if isinstance(data, np.ndarray) and data.shape == self.shape:
            super().__init__(data)
        else:
            raise ValueError("Input data must be a 3x3 NumPy array")

    def __repr__(self):
        return f"ThreeByThreeTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

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


class SixBySixTensor(Tensor):
    shape = (6, 6)

    def __init__(self, data):
        if isinstance(data, np.ndarray) and data.shape == self.shape:
            super().__init__(data)
        else:
            raise ValueError("Input data must be a 6x6 NumPy array")

    def __repr__(self):
        return f"SixBySixTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    def to_symmetric(self):
        pass


class SymmetricThreeByThreeTensor(Tensor):
    shape = (6, 1)

    def __init__(self, data):
        print(data.shape)
        if isinstance(data, np.ndarray) and data.shape == self.shape:
            self.data = data
        else:
            raise ValueError("Input data must be a 6x1 tensor")

    def __repr__(self):
        return f"SymmetricThreeByThreeTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    def to_general_tensor(self):
        components = [
            [self.data[0], self.data[5], self.data[4]],
            [self.data[5], self.data[1], self.data[3]],
            [self.data[4], self.data[3], self.data[2]],
        ]
        return ThreeByThreeTensor(np.array(components).reshape((3, 3)))
