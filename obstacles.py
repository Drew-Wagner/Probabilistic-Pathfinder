import numpy as np

from utils import segments_intersect


def point_cloud(point, r=0.1, n=1):
    points = [point]
    for i in range(n):
        points += Polygon.regular_polygon(point, r * (i + 1), 6 + (n - i))

    return points


class Obstacle:
    @property
    def segments(self):
        raise NotImplementedError()

    @property
    def vertices(self):
        raise NotImplementedError()


class Segment(Obstacle):
    def __init__(self, point_a, point_b):
        self.point_a = np.array(point_a)
        self.point_b = np.array(point_b)

    @property
    def vertices(self):
        points = []
        for point in [self.point_a, self.point_b]:
            points.extend(point_cloud(point))
        return points

    @property
    def segments(self):
        return [self]

    def plot(self, plt, color='r'):
        points = np.array([self.point_a, self.point_b])
        plt.plot(points[:, 0], points[:, 1], color=color)

    def intersects(self, points):
        a1 = self.point_a
        a2 = self.point_b
        if isinstance(points, Segment):
            b1 = points.point_a
            b2 = points.point_b
        else:
            b1, b2 = points

        return segments_intersect(a1, a2, b1, b2)


class Polygon(Obstacle):
    def __init__(self, points):
        self.points = [np.array(point) for point in points]
    
    @property
    def vertices(self):
        points = []
        for point in self.points:
            points.extend(point_cloud(point))
        return points

    @property
    def segments(self):
        segments = []

        n = len(self.points)
        for i, point_a in enumerate(self.points):
            point_b = self.points[(i + 1) % n]
            segments.append(Segment(point_a, point_b))
    
        return segments

    @staticmethod
    def regular_polygon(center, radius, sides):
        delta = 2 * np.pi / sides
        points = []

        for theta in np.arange(0, 2*np.pi, delta):
            point = np.array([np.cos(theta), np.sin(theta)]) * radius
            points.append(center + point)

        return points


class Circle (Polygon):
    def __init__(self, center, radius, n=12):
        self.center = np.array(center)
        self.radius = radius

        outer_radius = self.radius / np.cos(np.pi / n)
        super().__init__(self.regular_polygon(center, outer_radius, n))
    
