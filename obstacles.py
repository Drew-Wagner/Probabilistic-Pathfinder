import numpy as np


class Obstacle:
    @property
    def segments(self):
        raise NotImplemented

    @property
    def vertices(self):
        raise NotImplemented


class Segment(Obstacle):
    def __init__(self, point_a, point_b):
        self.point_a = point_a
        self.point_b = point_b

    @property
    def vertices(self):
        return [self.point_a, self.point_b]

    @property
    def segments(self):
        return [self]

    def plot(self, plt, color='r'):
        points = np.array([self.point_a, self.point_b])
        plt.plot(points[:, 0], points[:, 1], color=color)


class Polygon(Obstacle):
    def __init__(self, points):
        self.points = [np.array(point) for point in points]
    
    @property
    def vertices(self):
        return self.points

    @property
    def segments(self):
        segments = []

        n = len(self.points)
        for i, point_a in enumerate(self.points):
            point_b = self.points[(i + 1) % n]
            segments.append(Segment(point_a, point_b))

            opposite = self.points[(i + n // 2) % n]
            segments.append(Segment(point_a, opposite))
        
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
    
