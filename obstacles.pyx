# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False

import numpy as np
cimport numpy as np

from utils import segments_intersect, closestDistanceBetweenLines

from shapely.geometry import Polygon as SPolygon, LineString, Point

import shapely.prepared


cdef point_cloud((double, double) point_tuple, is_inside=lambda p: False, double radius=0.1, int n=6):
    cdef double delta, theta
    cdef np.ndarray d, p
    cdef list points
    cdef np.ndarray point = np.array(point_tuple)
    delta = 2 * np.pi / n

    points = []
    for theta in np.arange(0, 2*np.pi, delta):
        d = np.array([np.cos(theta), np.sin(theta)]) * radius
        p = point + d
        if not is_inside(p):
            points.append(p)

    return points


class Obstacle:
    @property
    def segments(self):
        raise NotImplementedError()

    @property
    def vertices(self):
        raise NotImplementedError()


class Vertex(Point):
    next_id = 0

    def __init__(self, *args):
        super().__init__(*args)
        self._id = Vertex.next_id
        Vertex.next_id += 1
        self.lines_of_sight = set()
        self.costs = {}

    def add_lines_of_sight(self, lines_of_sight):
        for line_of_sight, cost in lines_of_sight:
            self.lines_of_sight.add(line_of_sight)
            self.costs[line_of_sight] = cost
            if self not in line_of_sight.lines_of_sight:
                line_of_sight.add_lines_of_sight([(self, cost)])

    def remove_line_of_sight(self, line_of_sight):
        self.lines_of_sight.discard(line_of_sight)
        self.costs.pop(line_of_sight, None)

    def detach(self):
        while self.lines_of_sight:
            line_of_sight = self.lines_of_sight.pop()
            line_of_sight.remove_line_of_sight(self)

    def __hash__(self):
        return self._id


class Segment(LineString, Obstacle):    
    def __init__(self, *args):
        super().__init__(*args)

    @property
    def segments(self):
        return [self]
    
    @property
    def vertices(self):
        cdef list points
        points = []
        for point in self.coords:
            points.extend(point_cloud(point))
        return points

    def intersects(self, other):
        other = Segment(other)
        return super(LineString, self).intersects(other)

    def distance(self, other):
        other = Segment(other)
        return super(LineString, self).distance(other)

    def plot(self, plt, color='r'):
        points = np.array(self.coords)
        plt.plot(points[:, 0], points[:, 1], color=color)


class Polygon(SPolygon, Obstacle):
    def __init__(self, coordinates):
        super().__init__(coordinates)

    @property
    def segments(self):
        segments = []
        for i in range(len(self.exterior.coords) - 1):
            segments.append(
                Segment([self.exterior.coords[i], self.exterior.coords[i + 1]]))
        return segments


    @property
    def vertices(self):
        points = []
        for point in self.exterior.coords:
            points.extend(point_cloud(
                point, is_inside=lambda p: self.intersects(Point(p))))
        return points

    @staticmethod
    def regular_polygon(center, radius, sides):
        delta = 2 * np.pi / sides
        points = []

        for theta in np.arange(0, 2*np.pi, delta):
            point = np.array([np.cos(theta), np.sin(theta)]) * radius
            points.append(center + point)

        return points


class Circle (Polygon):
    def __init__(self, center, radius, n=10):
        self.center = np.array(center)
        self.radius = radius
        self.outer_radius = self.radius / np.cos(np.pi / n)
        super().__init__(self.regular_polygon(center, self.outer_radius, n))
