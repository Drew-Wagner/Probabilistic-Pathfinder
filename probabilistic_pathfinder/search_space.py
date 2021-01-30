import numpy as np

import rtree
from rtree.index import Property

from .obstacles import Obstacle

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

        if not self.is_free((a + b) / 2):
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

    def add_vertex(self, idx, point: np.ndarray, k=8):
        self.remove_vertex(idx)
        if not self._integrate_vertex(idx, point, k):
            raise Exception(f'Could not add vertex {point} ({idx}) to graph')

    def detach_vertex(self, idx):
        if idx not in self.edges:
            return

        for nidx in self.edges[idx]:
            self.edges[nidx].discard(idx)
        self.edges[idx].clear()

    def remove_vertex(self, idx):
        self.vertices.pop(idx, None)
        self.vertex_index.delete(idx, self.bounds)
        self.detach_vertex(idx)
        self.edges.pop(idx, None)

    def _integrate_vertex(self, idx, point, k=8):
        neighbours = list(self.vertex_index.nearest(tuple(point), k))
        connected = not bool(len(neighbours))

        self.edges[idx] = set()

        for neighbour_idx in neighbours:
            connected |= self._try_connect(idx, neighbour_idx, point=point)

        if connected:
            self.vertices[idx] = point
            self.vertex_index.insert(idx, tuple(point))
        else:
            self.edges.pop(idx)

        return connected

    def _try_connect(self, idx, nidx, point=None):
        if point is None:
            point = self.vertices[idx]
        neighbour = self.vertices[nidx]
        if not self.intersects(point, neighbour):
            self.edges[idx].add(nidx)
            self.edges[nidx].add(idx)
            return True
        else:
            return False

    def construct(self, n=100, k=8):
        # Uniform spacing
        # x = np.random.uniform(self.min[0], self.max[0], int(x))
        # y = np.random.uniform(self.min[1], self.max[1], int(y))
        # z = np.linspace(self.min[2], self.max[2], int(z))

        # X, Y, Z = np.meshgrid(x, y, z)
        # samples = np.dstack([X.ravel(), Y.ravel(), Z.ravel()])[0]
        # samples = list(self.free_samples(n))  # Random Uniform

        idx = 0
        iobst = 0
        while idx < n:
            # if idx > n:
            obst = self.obstacles[iobst % len(self.obstacles)]
            point = obst.sample()
            # else:
            #     point = self.free_samples(1)[0]
            if self.within(point):
                if self._integrate_vertex(idx, point, k):
                    idx += 1
            iobst += 1

    def plot(self, ax, obstacles=True, vertices=False, graph=True):
        if hasattr(self, 'bounds'):
            min_x, min_y, min_z = self.min
            max_x, max_y, max_z = self.max
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_zlim(min_z, max_z)

        if obstacles:
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

    def find_path(self, *points, plot=None):
        self.add_vertex(-1, points[0])
        for i, point in enumerate(points[1:]):
            self.add_vertex(-(i + 2), point)

        path = []
        for i in range(len(points) - 1):
            self._try_connect(-(i + 1), -(i + 2))
            partial_path = self._astar(-(i + 1), -(i + 2))
            if partial_path is None:
                print('NO PATH')
                return None
            partial_path = self.refine_path(partial_path)
            path.extend(partial_path)

        if plot is not None:
            self._plot_path(plot, path)

        return path

    def refine_path(self, path):
        path = list(path)

        i = 0
        while i < len(path) - 1:
            current = path[i]
            j = len(path) - 1
            while j > i:
                test = path[j]
                if self._try_connect(current, test):
                    del path[i + 1: j]
                j -= 1
            i += 1

        # Smooth along y-axis
        if len(path) > 2:
            a = self.vertices[path[0]]
            b = self.vertices[path[-1]]
            v = b - a
            l = np.linalg.norm(v)
            v /= l
            for i in range(1, len(path) - 1):
                current = path[i]
                previous_point = self.vertices[path[i - 1]]
                next_point = self.vertices[path[i + 1]]
                point = self.vertices[current]
                v1 = point - a
                v1 /= 2
                t = np.dot(v, v1)
                t = min(1, max(0, t))
                p = a + v * t * l
                adjusted_point = np.copy(point)
                adjusted_point[2] = p[2]
                its = 0
                while not self.is_free(adjusted_point) or \
                        self.intersects(
                        adjusted_point, previous_point) or \
                        self.intersects(adjusted_point, next_point):
                    adjusted_point = (point + adjusted_point) / 2
                    if its > 8:
                        adjusted_point = point
                        break
                    its += 1

                point[2] = adjusted_point[2]

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
        return (self.min <= point).all() and (point <= self.max).all()

    def sample(self):
        return np.random.uniform(
            low=self.min,
            high=self.max)
