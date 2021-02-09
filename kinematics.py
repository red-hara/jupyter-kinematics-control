import numpy as np


class Vector:
    def __init__(self, x, y, z):
        super(Vector, self).__init__()
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def zero():
        return Vector(0, 0, 0)

    def normalized(self):
        return 1 / self.magnitude() * self

    def magnitude(self):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

    def times(self, other):
        return Vector(
            other * self.x,
            other * self.y,
            other * self.z
        )

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
    def __init__(self, w, x, y, z):
        super(Quaternion, self).__init__()
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def identity():
        return Quaternion(1, 0, 0, 0)

    def rotate(self, vector):
        pure = Quaternion(0, vector.x, vector.y, vector.z)
        rotated = self * pure * self.conjugate()
        return Vector(rotated.x, rotated.y, rotated.z)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def multiply(self, other):
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        )

    def times(self, other):
        return Quaternion(
            other * self.w,
            other * self.x,
            other * self.y,
            other * self.z
        )

    @staticmethod
    def from_angle_axis(angle, axis):
        axis = axis.normalized()
        c = np.cos(angle / 2.0)
        s = np.sin(angle / 2.0)
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

    def __rmul__(self, other):
        return self.times(other)


class Transform:
    def __init__(self, translation: Vector, rotation: Quaternion):
        super(Transform, self).__init__()
        self.translation = translation
        self.rotation = rotation

    @staticmethod
    def identity():
        return Transform(
            Vector.zero(),
            Quaternion.identity()
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
