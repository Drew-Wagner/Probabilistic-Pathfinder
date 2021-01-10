import numpy as np

from utils import intersect

from collections import defaultdict
import heapq

from obstacles import Segment, Obstacle

class Vertex:
    def __init__(self, point):
        self.point = np.array(point)
        self.lines_of_sight = set()

    def add_lines_of_sight(self, lines_of_sight):
        self.lines_of_sight.update(lines_of_sight)
        for line_of_sight in lines_of_sight:
            if self not in line_of_sight.lines_of_sight:
                line_of_sight.add_lines_of_sight([self])

    def remove_line_of_sight(self, line_of_sight):
        self.lines_of_sight.discard(line_of_sight)

    def detach(self):
        while self.lines_of_sight:
            line_of_sight = self.lines_of_sight.pop()
            line_of_sight.remove_line_of_sight(self)

    def __repr__(self):
        return str(self.point)
        

class Pathfinder:
    def __init__(self, blocking_segments=None):
        self.blocking_segments = blocking_segments or set()
        self.vertices = set()
        self.start_node = None
        self.end_node = None
        self.path = None
        
    def add_obstacle(self, obstacle):
        if isinstance(obstacle, Obstacle):
            self.blocking_segments.update(obstacle.segments)

            for np_vertex in obstacle.vertices:
                vertex = Vertex(np_vertex)
                self.vertices.add(vertex)
                vertex.add_lines_of_sight(
                    self.find_lines_of_sight(vertex))
        else:
            raise ValueError(obstacle)

    def find_lines_of_sight(self, vertex):
        lines_of_sight = set()

        for possible_vertex in self.vertices:
            if possible_vertex == vertex:
                continue

            has_line_of_sight = True
            for segment in self.blocking_segments:
                if intersect(vertex.point, possible_vertex.point, segment.point_a, segment.point_b):
                    has_line_of_sight = False
                    break
                
            if has_line_of_sight:
                lines_of_sight.add(possible_vertex)
        
        return lines_of_sight

    def find_path(self, start_point=None, end_point=None):
        if start_point is not None:
            self.set_start_node(start_point)
        if end_point is not None:
            self.set_end_node(end_point)

        self.path = None
        if not (self.start_node and self.end_node):
            raise Exception('both start_node and end_node must be defined on the class or passed as parameters')

        self.path = self.a_star(self.start_node, self.end_node)
        
        return self.path

    def add_vertex(self, node):
        if not isinstance(node, Vertex):
            node = Vertex(node)

        self.vertices.add(node)
        node.add_lines_of_sight(self.find_lines_of_sight(node))
        return node

    def set_start_node(self, node):
        if self.start_node:
            self.start_node.detach()
        node = self.add_vertex(node)
        self.start_node = node

    def set_end_node(self, node):
        if self.end_node:
            self.end_node.detach()
        node = self.add_vertex(node)
        self.end_node = node

    def plot(self, plt, lines_of_sight=True, blocking_segments=True, path=True):
        if lines_of_sight:
            for v in self.vertices:
                for v2 in v.lines_of_sight:
                    points = np.array([v.point, v2.point])
                    plt.plot(points[:, 0], points[:, 1], '-', color='lightgrey')

        if blocking_segments:
            for segment in self.blocking_segments:
                segment.plot(plt, 'red')

        if path and self.path:
            path_points = np.array([ node.point for node in self.path ])
            plt.plot(path_points[:, 0], path_points[:, 1], '-', color='black')
            plt.plot(
                self.start_node.point[0], self.start_node.point[1], 'o', color='green')
            plt.plot(
                self.end_node.point[0], self.end_node.point[1], 'o', color='blue')
        
    @classmethod
    def reconstruct_path(cls, came_from, current):
        total_path = [ current ]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        
        return total_path

    @classmethod
    def a_star(cls, start, goal):
        def d (a, b):
            return np.linalg.norm(b.point - a.point)

        def h (node):
            return d(node, goal)

        g_score = defaultdict(lambda: np.Inf)
        g_score[start] = 0

        f_score = defaultdict(lambda: np.Inf)
        f_score[start] = h(start)
        
        class heap_node:
            def __init__(self, node):
                self.node = node
    
            def __cmp__(self, other):
                return f_score[other.node] - f_score[self.node]

            def __lt__(self, other):
                return f_score[self.node] < f_score[other.node]


        open_set = [ heap_node(start) ]
        heapq.heapify(open_set)

        came_from = { }

        while open_set:
            current = open_set[0].node
            if current == goal:
                return cls.reconstruct_path(came_from, current)
                    
            heapq.heappop(open_set)
            for neighbour in current.lines_of_sight:
                tentative_g_score = g_score[current] + d(current, neighbour)
                if tentative_g_score < g_score[neighbour]:
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g_score
                    f_score[neighbour] = g_score[neighbour] + h(neighbour)

                    if neighbour not in open_set:
                        heapq.heappush(open_set, heap_node(neighbour))

        return None


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from obstacles import Circle

    pf = Pathfinder()

    c = Circle((-0.1, 0), 0.25)
    l1 = Segment((-0.75, -0.5), (-0.4, 0.8))
    l2 = Segment((-0.4, 0.8), (0.5, 0.3))
    pf.add_obstacle(c)
    pf.add_obstacle(l1)
    pf.add_obstacle(l2)

    start_point = np.array((-1, 0))
    end_point = np.array((0.4, 0.1))

    path = pf.find_path(start_point, end_point)
    pf.plot(plt)

    plt.gca().set_aspect('equal')
    plt.savefig('./images/example.png')
    plt.show()
