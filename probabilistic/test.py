from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

import rtree
from rtree.index import Property

from obstacles import Obstacle, Cylinder

import heapq
from collections import defaultdict


class SearchSpace:
    def __init__(self):
        self.obstacles = {}
        self.obstacle_index = rtree.index.Index(
            properties=Property(dimension=3))
        self.vertices = {}
        self.edges = {}
        self.vertex_index = rtree.index.Index(
            properties=Property(dimension=3))

    def insert_obstacle(self, obstacle: Obstacle):
        id_ = getattr(self, '_next_id', 0)
        self._next_id = id_ + 1

        self.obstacles[id_] = obstacle
        self.obstacle_index.insert(id_, obstacle.bounds)
        return id_

    def within(self, point: np.ndarray):
        raise NotImplementedError()

    def is_free(self, point: np.ndarray):
        if not self.within(point):
            return False

        possible_obstacles = self.obstacle_index.intersection(tuple(point))
        for oid in possible_obstacles:
            obstacle = self.obstacles[oid]
            if obstacle.within(point):
                return False

        return True

    def intersects(self, a: np.ndarray, b: np.ndarray):
        a, b = np.array(a), np.array(b)
        b_minus_a = b - a
        if self.within(b_minus_a / 2):
            return True

        min_x, max_x = min(a[0], b[0]), max(a[0], b[0])
        min_y, max_y = min(a[1], b[1]), max(a[1], b[1])
        min_z, max_z = min(a[2], b[2]), max(a[2], b[2])
        bounds = (min_x, min_y, min_z, max_x, max_y, max_z)
        possible_obstacles = self.obstacle_index.intersection(bounds)
        for oid in possible_obstacles:
            obstacle = self.obstacles[oid]
            if obstacle.intersects(a, b):
                return True

        return False

    def sample(self, n=1):
        raise NotImplementedError()

    def free_samples(self, n: int = 1):
        samples = np.empty((n, 3))
        sample_count = 0
        while sample_count < n:
            sample = self.sample()
            if self.is_free(sample):
                samples[sample_count] = sample
                sample_count += 1
        return samples

    def _insert_vertex(self, idx, point: np.ndarray):
        self.vertices[idx] = point
        self.edges[idx] = set()
        self.vertex_index.insert(idx, tuple(point))

    def add_vertex(self, idx, point: np.ndarray):
        self.remove_vertex(idx)
        self._insert_vertex(idx, point)
        self._integrate_vertex(idx)

    def detach_vertex(self, idx):
        if idx not in self.edges:
            return

        for nidx in self.edges[idx]:
            self.edges[nidx].discard(idx)
        self.edges[idx].clear()

    def remove_vertex(self, idx):
        self.detach_vertex(idx)
        self.vertices.pop(idx, None)
        self.edges.pop(idx, None)

    def _integrate_vertex(self, idx, k=8):
        point = self.vertices[idx]
        neighbours = self.vertex_index.nearest(tuple(point), k)
        for neighbour_idx in neighbours:
            if neighbour_idx not in self.edges[idx]:
                self._try_connect(idx, neighbour_idx, point=point)

    def _try_connect(self, idx, nidx, point=None):
        if point is None:
            point = self.vertices[idx]
        neighbour = self.vertices[nidx]
        if not self.intersects(point, neighbour):
            self.edges[idx].add(nidx)
            self.edges[nidx].add(idx)

    def _construct_nodes(self, n):
        samples = self.free_samples(n)  # Random Uniform

        # # Uniform spacing
        # x = np.linspace(self.min[0], self.max[0], int(n**(1/3)))
        # y = np.linspace(self.min[1], self.max[1], int(n**(1/3)))
        # z = np.linspace(self.min[2], self.max[2], int(n**(1/3)))

        # X, Y, Z = np.meshgrid(x, y, z)
        # samples = np.dstack([X.ravel(), Y.ravel(), Z.ravel()])[0]
        for idx, point in enumerate(samples):
            self._insert_vertex(idx, point)

    def construct(self, n=500, k=8):
        self._construct_nodes(n)

        for idx in self.vertices:
            self._integrate_vertex(idx, k=k)

    def plot(self, ax, vertices=False, graph=True):
        if hasattr(self, 'bounds'):
            min_x, min_y, min_z = self.min
            max_x, max_y, max_z = self.max
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_zlim(min_z, max_z)

        for obstacle in self.obstacles.values():
            obstacle.plot(ax)

        if graph:
            drawn_edges = set()
            for idx, edges in self.edges.items():
                point = self.vertices[idx]
                for nidx in edges:
                    if frozenset((idx, nidx)) in drawn_edges:
                        continue
                    npoint = self.vertices[nidx]
                    points = np.array([point, npoint])
                    ax.plot(points[:, 0], points[:, 1],
                            points[:, 2], color='grey')
                    drawn_edges.add(frozenset((idx, nidx)))
        if vertices:
            points = np.array(list(self.vertices.values()))
            ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    def distance(self, aidx, bidx):
        point_a = self.vertices[aidx]
        point_b = self.vertices[bidx]
        return np.linalg.norm(point_b - point_a)

    def _astar(self, sidx, gidx):
        """
        Based on pseudocode from:
        https://en.wikipedia.org/wiki/A*_search_algorithm
        """

        g_score = defaultdict(lambda: np.Inf)
        g_score[sidx] = 0

        f_score = defaultdict(lambda: np.Inf)
        f_score[sidx] = self.distance(sidx, gidx)

        class heap_node:
            def __init__(self, idx):
                self.idx = idx

            def __cmp__(self, other):
                return f_score[other.idx] - f_score[self.idx]

            def __lt__(self, other):
                return f_score[self.idx] < f_score[other.idx]

        open_set = [heap_node(sidx)]
        heapq.heapify(open_set)

        came_from = {}

        while open_set:
            current = open_set[0].idx
            if current == gidx:
                break

            heapq.heappop(open_set)
            for neighbour in self.edges[current]:
                tentative_g_score = g_score[current] + \
                    self.distance(current, neighbour)
                if tentative_g_score < g_score[neighbour]:
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g_score
                    f_score[neighbour] = g_score[neighbour] + \
                        self.distance(neighbour, gidx)

                    if neighbour not in open_set:
                        heapq.heappush(open_set, heap_node(neighbour))
        else:
            return None

        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)

        return total_path

    def find_path(self, point_a, *points, plot=None):
        self.add_vertex(-1, point_a)
        for i, point in enumerate(points):
            self.add_vertex(-(i + 2), point)

        path = []
        for i in range(len(points)):
            self._try_connect(-(i + 1), -(i + 2))
            partial_path = self._astar(-(i + 1), -(i + 2))
            if partial_path is None:
                return None
            partial_path = self.refine_path(partial_path)
            path.extend(partial_path)

        if plot is not None:
            self._plot_path(plot, path)

        return path

    def refine_path(self, path):
        path = list(path)
        popped = True
        while popped:
            popped = False
            current_idx = 0
            while current_idx < len(path) - 2:
                current = path[current_idx]
                next_ = path[current_idx + 2]
                point_a = self.vertices[current]
                point_b = self.vertices[next_]

                if not self.intersects(point_a, point_b):
                    popped = True
                    path.pop(current_idx + 1)
                else:
                    current_idx += 1

        return path

    def _plot_path(self, ax, path):
        points = np.array([
            self.vertices[p] for p in path
        ])
        ax.plot(points[:, 0], points[:, 1], points[:, 2])
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])


class BoxSearchSpace(SearchSpace):

    def __init__(self, bounds: np.ndarray = (0, 0, 0, 1, 1, 1)):
        super().__init__()
        self.bounds = np.array(bounds)
        self.min, self.max = self.bounds[:3], self.bounds[3:]

    def within(self, point: np.ndarray):
        return (self.min <= point).all() and (point < self.max).all()

    def sample(self):
        return np.random.uniform(
            low=self.min,
            high=self.max)


if __name__ == "__main__":
    import time

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    box = BoxSearchSpace()

    for i in range(10):
        p = np.random.uniform(size=(3,))
        p[2] = 0
        cyl = Cylinder(p, np.random.uniform(low=0.01, high=0.2),
                    np.random.uniform(low=0.05))
        box.insert_obstacle(cyl)

    t = time.time()
    box.construct(n=500)
    print("Time to construct", time.time() - t)

    a = np.array((0, 0, 0.5))
    b = np.array((0.5, 0.5, 0.5))
    c = np.array((0, 1, 0.5))
    box.plot(ax, graph=False, vertices=False)

    t = time.time()
    path = box.find_path(a, b, c, plot=ax)
    print("Time to find path", time.time() - t)
    plt.show()
