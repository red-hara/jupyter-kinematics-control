import numpy as np


class Vector:
    """Vector represents translation and stores 3 values

    Attributes
    ----------
    x
        The x value
    y
        The y value
    z
        The z value
    """

    def __init__(self, x, y, z):
        """
        Parameters
        ----------
        x
            The x value
        y
            The y value
        z
            The z value
        """

        super(Vector, self).__init__()
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def zero():
        """Creates vector with zero x, y and z"""

        return Vector(0, 0, 0)

    @staticmethod
    def lerp(start, end, t):
        """Calculates linear interpolation between start and end for given t"""

        return start + t * (end + -1 * start)

    def normalized(self):
        """Calculates colinear vector with length 1"""

        return 1 / self.magnitude() * self

    def magnitude(self):
        """Calculates Euclidean length of vector"""

        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def times(self, other):
        """Multiplies vector by scalar value

        Parameters
        ----------
        other
            The scalar value to multiply on
        """

        return Vector(
            other * self.x,
            other * self.y,
            other * self.z
        )

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def angle_to(self, other, axis):
        crossed = self.cross(other)
        collinearity = crossed.dot(axis)
        return np.arctan2(
            crossed.magnitude(),
            self.dot(other)
        ) * (1 if collinearity > 0.0 else -1)

    def __str__(self):
        return "[{}, {}, {}]".format(self.x, self.y, self.z)

    def __repr__(self):
        return "Vector[{}, {}, {}]".format(self.x, self.y, self.z)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(
                self.x + other.x,
                self.y + other.y,
                self.z + other.z
            )

    def __mul__(self, other):
        return self.times(other)

    def __rmul__(self, other):
        return self.times(other)


class Quaternion:
    """Quaternion represents rotation and stores 4 values

    Attributes
    ----------
    w
        The w value
    x
        The x value
    y
        The y value
    z
        The z value
    """

    def __init__(self, w, x, y, z):
        """
        Parameters
        ----------
        w
            The w value
        x
            The x value
        y
            The y value
        z
            The z value
        """

        super(Quaternion, self).__init__()
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def identity():
        """Creates quaternion representing no rotation"""

        return Quaternion(1, 0, 0, 0)

    def rotate(self, vector):
        """Rotates given vector

        Parameters
        ----------
        vector
            Vector to be rotated
        """

        pure = Quaternion(0, vector.x, vector.y, vector.z)
        rotated = self * pure * self.conjugate()
        return Vector(rotated.x, rotated.y, rotated.z)

    def magnitude(self):
        return (self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def normalized(self):
        return 1 / self.magnitude() * self

    def dot(self, other):
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    @staticmethod
    def slerp(start, end, t):
        dot = start.dot(end)
        if dot < 0.0:
            start = -1 * start
            dot = -dot
        if dot >= 1.0:
            return start
        theta = np.arccos(dot) * t
        q = (end + -dot * start).normalized()
        return np.cos(theta) * start + np.sin(theta) * q

    def conjugate(self):
        """Calculates quaternion representing reverse rotation"""

        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def multiply(self, other):
        """Multiplies quaternion on other quaternion

        Parameters
        ----------
        other
            The other quaternion to multiply on
        """

        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )

    def plus(self, other):
        return Quaternion(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def times(self, other):
        """Multiplies quaternion on scalar

        Parameters
        ----------
        other
            The scalar value to multiply on
        """

        return Quaternion(
            other * self.w,
            other * self.x,
            other * self.y,
            other * self.z
        )

    @staticmethod
    def from_angle_axis(angle, axis, math_source=np, unit=True):
        """Calculates quaternion representing rotation around given axis on given angle

        Parameters
        ----------
        angle
            The rotation angle in radians
        axis
            The rotation axis
        math_source
            The source of sin and cos operations, default is numpy
        unit
            Marks if axis is considered unit, default is True
        """

        if not unit:
            axis = axis.normalized()
        c = math_source.cos(angle / 2)
        s = math_source.sin(angle / 2)
        return Quaternion(
            c,
            s * axis.x,
            s * axis.y,
            s * axis.z
        )

    def __str__(self):
        return "[{}, {}, {}, {}]".format(self.w, self.x, self.y, self.z)

    def __repr__(self):
        return "Quaternion[{}, {}, {}, {}]".format(self.w, self.x, self.y, self.z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.rotate(other)
        if isinstance(other, Quaternion):
            return self.multiply(other)
        return self.times(other)

    def __add__(self, other):
        return self.plus(other)

    def __rmul__(self, other):
        return self.times(other)


class Transform:
    """Transform represents translation and rotation

    Attributes
    ----------
    translation
        The translation vector
    rotation
        The rotation quaternion
    """

    def __init__(self, translation: Vector, rotation: Quaternion):
        """
        Parameters
        ----------
        translation
            The translation vector
        rotation
            The rotation quaternion
        """

        super(Transform, self).__init__()
        self.translation = translation
        self.rotation = rotation

    @staticmethod
    def identity():
        """Creates Transform representing no translation and no rotation"""

        return Transform(
            Vector.zero(),
            Quaternion.identity()
        )

    def inverse(self):
        conj = self.rotation.conjugate()
        return Transform(
            conj * (-1 * self.translation),
            conj
        )

    @staticmethod
    def lerp(start, end, t):
        return Transform(
            Vector.lerp(start.translation, end.translation, t),
            Quaternion.slerp(start.rotation, end.rotation, t)
        )

    def __str__(self):
        return "[\n\t{}\n\t{}\n]".format(str(self.translation), str(self.rotation))

    def __repr__(self):
        return "Transform[\n\t{}\n\t{}\n]".format(str(self.translation), str(self.rotation))

    def __add__(self, other):
        if isinstance(other, Transform):
            return Transform(
                self.translation + self.rotation * other.translation,
                self.rotation * other.rotation
            )
        if isinstance(other, Vector):
            return self.translation + self.rotation * other
