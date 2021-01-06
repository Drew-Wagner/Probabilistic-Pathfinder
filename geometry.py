import numpy as np


# Generate a circumscribing polygon around a circle
def regular_polygon(center, radius, sides):
    delta = 2 * np.pi / sides
    points = []

    for theta in np.arange(0, 2*np.pi, delta):
        point = np.array([np.cos(theta), np.sin(theta)]) * radius
        points.append(center + point)

    return np.array(points)


def get_centroid(points):
    return np.mean(points, axis=0)
