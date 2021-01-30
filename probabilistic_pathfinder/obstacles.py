import numpy as np

from .intersections import moller_trumbore_ray_triangle_intersects


class Obstacle:
    def __init__(self, bounds: np.ndarray):
        self.bounds = np.array(bounds)
        self.min, self.max = self.bounds[:3], self.bounds[3:]
        self.triangles = []
        self.vertices = []

    def sample(self):
        raise NotImplementedError()

    def within(self, point: np.ndarray):
        if not isinstance(point, np.ndarray):
            point = np.array(point)

        return (self.min <= point).all() and (point < self.max).all()

    def intersects(self, a: np.ndarray, b: np.ndarray):
        a, b = np.array(a), np.array(b)
        b_minus_a = b - a
        length = np.linalg.norm(b_minus_a)
        if self.within(b_minus_a / 2):
            return True

        for triangle in self.triangles:
            triangle = self.vertices[triangle]
            *_, dist = moller_trumbore_ray_triangle_intersects(
                a, b_minus_a, triangle)
            if dist is not None and dist <= length:
                return True

        return False

    def plot(self, ax, wireframe=True):
        if wireframe:
            for triangle in self.triangles:
                points = self.vertices[triangle]
                np.append(points, points[0])
                ax.plot(points[:, 0], points[:, 1], points[:, 2], color='0.2')

        else:
            ax.plot_trisurf(self.vertices[:, 0],
                            self.vertices[:, 1],
                            self.vertices[:, 2],
                            triangles=self.triangles)
        ax.scatter3D(
            self.vertices[:, 0],
            self.vertices[:, 1],
            self.vertices[:, 2])


class Cylinder(Obstacle):
    def __init__(self, point: np.ndarray, radius: float, height: float, sides: int = 12):
        self.point = np.array(point)
        self.radius = radius
        self.rsquared = radius * radius
        self.height = height
        self.sides = sides

        self.min = self.point - self.radius
        self.max = self.point + self.radius + np.array((0, 0, self.height))
        super().__init__((*self.min, *self.max))

        self.calculate_triangles()

    def sample(self):
        angle = np.random.uniform(0, 2 * np.pi)
        height = np.random.uniform(
            0, self.height) + np.abs(np.random.normal(0, self.radius))
        if height <= self.height:
            radius = self.radius + \
                np.abs(np.random.normal(0, self.radius))
        else:
            dh = height - self.height
            radius = (self.radius**2 - dh**2) ** 0.5

        point = np.zeros((3,))
        point[0] = radius * np.cos(angle) + self.point[0]
        point[1] = radius * np.sin(angle) + self.point[1]
        point[2] = height
        return point

    def within(self, point):
        if not isinstance(point, np.ndarray):
            point = np.array(point)

        if not super().within(point):
            return False

        point_xy = point[:2]
        base_xy = self.point[:2]
        diff = point_xy - base_xy
        return np.dot(diff, diff) < self.rsquared

    def calculate_triangles(self):
        triangles = []

        approximated_radius = self.radius / np.cos(np.pi / self.sides)
        base_circle_points = regular_polygon(
            self.point, approximated_radius, self.sides)
        top_circle_points = np.copy(base_circle_points)
        top_circle_points[:, 2] += self.height

        self.vertices = np.vstack((base_circle_points, top_circle_points))

        len_circle = len(base_circle_points)
        for i in range(self.sides):
            # Add sides
            triangleA = [
                i,
                (i + 1) % len_circle + len_circle,
                (i + 1) % len_circle,
            ]
            triangleB = [
                i,
                i + len_circle,
                (i + 1) % len_circle + len_circle,
            ]
            triangles.append(triangleA)
            triangles.append(triangleB)

            # Add caps
            if i + 2 < len_circle:
                triangleC = [
                    0,
                    i + 1,
                    i + 2
                ]
                triangleD = [
                    len_circle,
                    i + 1 + len_circle,
                    i + 2 + len_circle
                ]
                triangles.append(triangleC)
                triangles.append(triangleD)

        self.triangles = np.array(triangles)


def regular_polygon(center, radius, sides):
    delta = 2 * np.pi / sides
    points = []

    for theta in np.arange(0, 2*np.pi, delta):
        point = np.array([np.cos(theta), np.sin(theta), 0]) * radius
        points.append(center + point)

    return np.array(points)
