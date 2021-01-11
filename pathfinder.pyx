# cython: language_level=3, boundscheck=False, cdivisio=True

import numpy as np
cimport numpy as np

from collections import defaultdict
import heapq

from obstacles import Segment, Obstacle, Vertex

import rtree
import time

from scipy.spatial import KDTree


cdef close_pass_cost(double distance):
    cdef double cost = np.Inf
    if distance != 0:
        cost = max(0, -np.log(distance * 10))
    return cost


class Pathfinder:
    def __init__(self):
        self.blocking_segments = []
        self.segment_index = rtree.index.Index()
        self.vertices = []
        self.kd_tree = None
        self.start_node = None
        self.end_node = None
        self.path = None

    def add_obstacle(self, obstacle):
        if isinstance(obstacle, Obstacle):
            t = time.time()
            segments = obstacle.segments
            for segment in segments:
                self.segment_index.insert(len(self.blocking_segments), segment.bounds)
                self.blocking_segments.append(segment)

            for np_vertex in obstacle.vertices:
                vertex = Vertex(np_vertex)
                self.add_vertex(vertex)
            print('Time to add obstacle: ', time.time() - t)

        else:
            raise ValueError(obstacle)

    def find_lines_of_sight(self, vertex):
        cdef set lines_of_sight = set()
        cdef double t = time.time()
        cdef int vertices_considered = 0
        cdef double closest_pass
        cdef int has_line_of_sight
        cdef double distance
        cdef int index

        for possible_vertex in self.vertices:
            if possible_vertex == vertex or vertex in possible_vertex.lines_of_sight:
                continue
            vertices_considered += 1
            has_line_of_sight = True
            closest_pass = np.Inf

            path = Segment([vertex, possible_vertex])
            for index in self.segment_index.intersection(path.bounds):
                segment = self.blocking_segments[index]

                if segment.intersects(path):
                    has_line_of_sight = False
                    break
                else:
                    distance = max(0, segment.distance(path))
                    closest_pass = min(closest_pass, distance)

            if has_line_of_sight:
                cost = close_pass_cost(closest_pass)
                lines_of_sight.add((possible_vertex, cost))
            else:
                possible_vertex.remove_line_of_sight(vertex)

        return lines_of_sight

    def find_path(self, start_point=None, end_point=None):
        if start_point is not None:
            self.set_start_node(start_point)
        if end_point is not None:
            self.set_end_node(end_point)

        self.path = None
        if not (self.start_node and self.end_node):
            raise Exception(
                'both start_node and end_node must be defined on the class'
                ' or passed as parameters')

        t = time.time()
        self.path = self.a_star(self.start_node, self.end_node)
        print('Time to find path: ', time.time() - t)
        return self.path

    def add_vertex(self, node):
        if not isinstance(node, Vertex):
            node = Vertex(node)

        point = node.coords[0]
        
        if self.kd_tree is not None:
            d, i = self.kd_tree.query(point)
            if np.isclose(d, 0):
                # same point
                self.vertices[i].detach()
                self.vertices.pop(i)

        self.vertices.append(node)
        self.kd_tree = KDTree([v.coords[0] for v in self.vertices])
        node.add_lines_of_sight(self.find_lines_of_sight(node))
        return node

    def set_start_node(self, node):
        if self.start_node:
            try:
                self.vertices.remove(self.start_node)
            except ValueError:
                pass
            self.start_node.detach()
        node = self.add_vertex(node)
        self.start_node = node

    def set_end_node(self, node):
        if self.end_node:
            try:
                self.vertices.remove(self.end_node)
            except ValueError:
                pass
            self.end_node.detach()
        node = self.add_vertex(node)
        self.end_node = node

    def plot(self, plt, lines_of_sight=True,
             blocking_segments=True, path=True):
        if lines_of_sight:
            for v in self.vertices:
                for v2 in v.lines_of_sight:
                    points = np.array([v.coords[0], v2.coords[0]])
                    plt.plot(points[:, 0], points[:, 1],
                             '-', color='lightgrey')

        if blocking_segments:
            for segment in self.blocking_segments:
                segment.plot(plt, 'red')

        if path and self.path:
            path_points = np.array([p.coords[0] for p in self.path])
            plt.plot(
                path_points[:, 0], path_points[:, 1],
                '-', color='blue')

        if self.start_node:
            plt.plot(
                self.start_node.coords[0][0], self.start_node.coords[0][1],
                'o', color='green')
        if self.end_node:
            plt.plot(
                self.end_node.coords[0][0], self.end_node.coords[0][1],
                'o', color='blue')

    @classmethod
    def a_star(cls, start, goal):
        """
        Based on pseudocode from:
        https://en.wikipedia.org/wiki/A*_search_algorithm
        """
        g_score = defaultdict(lambda: np.Inf)
        g_score[start] = 0

        f_score = defaultdict(lambda: np.Inf)
        f_score[start] = start.distance(goal)

        class heap_node:
            def __init__(self, node):
                self.node = node

            def __cmp__(self, other):
                return f_score[other.node] - f_score[self.node]

            def __lt__(self, other):
                return f_score[self.node] < f_score[other.node]

        open_set = [heap_node(start)]
        heapq.heapify(open_set)

        came_from = {}

        while open_set:
            current = open_set[0].node
            if current == goal:
                break

            heapq.heappop(open_set)
            for neighbour in current.lines_of_sight:
                cost = current.costs[neighbour]
                tentative_g_score = g_score[current] + current.distance(neighbour) + cost
                if tentative_g_score < g_score[neighbour]:
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g_score
                    f_score[neighbour] = g_score[neighbour] + neighbour.distance(goal)

                    if neighbour not in open_set:
                        heapq.heappush(open_set, heap_node(neighbour))
        else:
            return None

        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)

        return total_path


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from obstacles import Circle

    t = time.time()
    pf = Pathfinder()

    obstacles = [
        Circle((-0.4, 0.12), 0.25),
        Circle((0.4, -0.12), 0.25),
    ]

    start_point = np.array((-1, 0))
    end_point = np.array((1, 0))

    for o in obstacles:
        pf.add_obstacle(o)

    path = pf.find_path(start_point, end_point)
    
    print('Total time: ', time.time() - t)
    pf.plot(plt, lines_of_sight=False)

    plt.gca().set_aspect('equal')
    plt.show()
