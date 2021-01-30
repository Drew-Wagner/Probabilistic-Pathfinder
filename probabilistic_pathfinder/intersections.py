import numpy as np


def moller_trumbore_ray_triangle_intersects(
        ray_origin, ray_direction, triangle, eps=0.0000001):
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    vertex0, vertex1, vertex2 = triangle

    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0

    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if a > -eps and a < eps:
        return False, None, None

    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, None, None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return False, None, None

    t = f * np.dot(edge2, q)
    if t > eps:
        intersection_point = ray_origin + ray_direction * t
        return True, intersection_point, t
    else:
        return False, None, None
