import numpy as np


class CoordSystem:
    def __init__(self, origin=np.zeros(3), basis_vectors=None):
        self.origin = np.array(origin)
        if basis_vectors is None:
            self.basis_vectors = np.eye(3)
        else:
            self.basis_vectors = np.array(basis_vectors)

    def __repr__(self):
        return f"CoordSystem(origin={self.origin}, basis_vectors={self.basis_vectors})"

    def to_local(self, global_coords):
        return np.linalg.solve(self.basis_vectors, global_coords - self.origin)

    def to_global(self, local_coords):
        return np.dot(self.basis_vectors, local_coords) + self.origin

class CartesianCoord(CoordSystem):
    def __init__(self, origin=np.zeros(3), x_axis=np.array([1, 0, 0]), y_axis=np.array([0, 1, 0])):
        # Set up the Cartesian coordinate system with the given axis directions
        z_axis = np.cross(x_axis, y_axis)
        basis_vectors = np.column_stack((x_axis, y_axis, z_axis))
        
        super().__init__(origin=origin, basis_vectors=basis_vectors)

    def __repr__(self):
        return f"CartesianCoordSystem(origin={self.origin}, basis_vectors={self.basis_vectors})"

class CylindricalCoord(CoordSystem):
    def __init__(self, origin=np.zeros(3), axis_direction=np.array([0, 0, 1])):
        z_axis = axis_direction / np.linalg.norm(axis_direction)
        x_axis = np.cross(z_axis, [0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)

        basis_vectors = np.column_stack((x_axis, y_axis, z_axis))

        super().__init__(origin=origin, basis_vectors=basis_vectors)

    def __repr__(self):
        return f"CylindricalCoordSystem(origin={self.origin}, basis_vectors={self.basis_vectors})"


class SphericalCoord(CoordSystem):
    def __init__(self, origin=np.zeros(3)):
        # Set up the spherical coordinate system
        # Here, we can use the standard spherical coordinates basis vectors
        basis_vectors = np.eye(3)

        super().__init__(origin=origin, basis_vectors=basis_vectors)

    def __repr__(self):
        return f"SphericalCoordSystem(origin={self.origin}, basis_vectors={self.basis_vectors})"
