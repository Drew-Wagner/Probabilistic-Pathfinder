import numpy as np

from geometry import regular_polygon

class Obstacle:

    def to_points(self):
        raise NotImplemented

    def is_inside(self, point):
        raise NotImplemented


class Circle (Obstacle):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
    
    def _circumscribed_polygon(self, sides):
        outer_radius = self.radius / np.cos(np.pi / sides)
        points = regular_polygon(self.center, outer_radius, sides)

        return points

    def is_inside(self, point):
        return np.linalg.norm(self.center - point) <= self.radius

    def to_points(self):
        return self._circumscribed_polygon(12)


def points_from_obstacles(obstacles):
    points = []
    for obs in obstacles:
        points.extend(obs.to_points())
    
    return points
