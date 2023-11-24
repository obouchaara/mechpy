import sympy as sp


class SymbolicTensor:
    def __init__(self, data):
        if isinstance(data, sp.Matrix):
            self.data = data
        else:
            raise ValueError("Input data must be a SymPy Matrix")

    def __repr__(self):
        return f"SymbolicTensor(\n{self.data}\n)"

    def __matmul__(self, other):
        pass

    def is_symmetric(self):
        # if isinstance(self, SymbolicSymmetricThreeByThreeTensor):
        #     return True
        return self.data == self.data.transpose()

    def is_second_rank(self):
        return True

    def is_fourth_rank(self):
        return False

    def is_3x3(self):
        # if isinstance(self, SymbolicSymmetricThreeByThreeTensor):
        #     return True
        return self.data.shape == (3, 3)

    def is_6x6(self):
        return self.data.shape == (6, 6)

    def is_3x3x3x3(self):
        return False

    def to_3x3(self):
        if self.is_3x3():
            return SymbolicThreeByThreeTensor(self.data)
        raise ValueError("The tensor is not a 3x3 matrix.")

    def to_sym_3x3(self):
        if self.is_3x3() and self.is_symmetric():
            return SymbolicSymmetricThreeByThreeTensor(self.data)
        raise ValueError("The tensor is not a symmetric 3x3 matrix.")

    def to_6x6(self):
        if self.is_6x6():
            return SymbolicSixBySixTensor(self.data)
        raise ValueError("The tensor is not a 6x6 matrix.")

    @classmethod
    def from_list(cls, components, shape, symmetric=False, mode=1):
        if isinstance(components, list):
            if symmetric:
                if shape == (3, 3) and mode == 1:
                    if len(components) != shape[0] * (shape[1] + 1) // 2:
                        raise ValueError("Invalid components for symmetric tensor")
                    # Construct a symmetric 3x3 matrix
                    data = sp.Matrix(
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
                    # Fill a symmetric matrix based on components
                    data = sp.zeros(*shape)
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
                    # Create a matrix with the given shape
                    data = sp.Matrix(components).reshape(*shape)
                except:
                    raise ValueError("Invalid components or shape value")
                return cls(data)
            else:
                raise ValueError("Invalid shape parameter")
        else:
            raise ValueError("Input must be a list")

    def __getitem__(self, key):
        return self.data[key]


class SymbolicThreeByThreeTensor(SymbolicTensor):
    shape = (3, 3)

    def __init__(self, data):
        if isinstance(data, sp.Matrix) and data.shape == self.shape:
            super().__init__(data)
        else:
            raise ValueError("Input data must be a 3x3 SymPy Matrix")

    def __repr__(self):
        return f"SymbolicThreeByThreeTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    def to_symmetric(self):
        if self.is_symmetric():
            IVM = SymbolicSymmetricThreeByThreeTensor.INVERSE_VOIGT_MAPPING
            data = sp.Matrix([self.data[IVM[i]] for i in range(6)])
            return SymbolicSymmetricThreeByThreeTensor(data)
        raise ValueError(
            "The tensor is not symmetric, and it cannot be converted to a symmetric tensor."
        )


class SymbolicSixBySixTensor(SymbolicTensor):
    shape = (6, 6)

    def __init__(self, data):
        if isinstance(data, sp.Matrix) and data.shape == self.shape:
            super().__init__(data)
        raise ValueError("Input data must be a 6x6 SymPy Matrix")

    def __repr__(self):
        return f"SymbolicSixBySixTensor(\n{self.data}\n)"

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)


class SymbolicSymmetricThreeByThreeTensor(SymbolicTensor):
    shape = (6, 1)

    VOIGT_MAPPING = {
        (0, 0): 0,
        (1, 1): 1,
        (2, 2): 2,
        (1, 2): 3,
        (2, 1): 3,
        (0, 2): 4,
        (2, 0): 4,
        (0, 1): 5,
        (1, 0): 5,
    }

    INVERSE_VOIGT_MAPPING = {
        0: (0, 0),
        1: (1, 1),
        2: (2, 2),
        3: (1, 2),
        4: (0, 2),
        5: (0, 1),
    }

    def __init__(self, data):
        if isinstance(data, sp.Matrix) and data.shape == self.shape:
            super().__init__(data)
        else:
            raise ValueError("Input data must be a 6x1 SymPy Matrix")

    def __repr__(self):
        return f"SymmetricSymbolicThreeByThreeTensor(\n{self.data}\n)"

    def is_symmetric(self):
        return True

    @classmethod
    def from_list(cls, components):
        return super().from_list(components, cls.shape)

    @classmethod
    def create(cls, name):
        VM = cls.INVERSE_VOIGT_MAPPING
        components = [sp.symbols(f"{name}_{i+1}{j+1}") for i, j in VM.values()]
        return cls.from_list(components)

    def to_general_tensor(self):
        VM = self.VOIGT_MAPPING
        components = [[self.data[VM[(i, j)]] for i in range(3)] for j in range(3)]

        data = sp.Matrix(components)
        return SymbolicThreeByThreeTensor(data)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, tuple) and len(key) == 2:
            return self.data[self.VOIGT_MAPPING[key]]
        else:
            raise ValueError("Key must be int or tuple of 2 elements")
