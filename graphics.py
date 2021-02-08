from kinematics import Vector


def chain_to_points(chain):
    x = []
    y = []
    z = []
    for transform in chain:
        x += [transform.translation.x]
        y += [transform.translation.y]
        z += [transform.translation.z]
    return (x, y, z)


def axis(ax, transform, size=1):
    origin = transform.translation
    x = transform + Vector(size, 0, 0)
    y = transform + Vector(0, size, 0)
    z = transform + Vector(0, 0, size)
    r, = ax.plot([origin.x, x.x], [origin.y, x.y], [origin.z, x.z], color="#ff0000")
    g, = ax.plot([origin.x, y.x], [origin.y, y.y], [origin.z, y.z], color="#00ff00")
    b, = ax.plot([origin.x, z.x], [origin.y, z.y], [origin.z, z.z], color="#0000ff")
    return (r, g, b)
