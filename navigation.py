from obstacles import points_from_obstacles
from geometry import get_centroid

from scipy.spatial import Delaunay, delaunay_plot_2d

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import numpy as np

class NavMesh(Delaunay):

    def __init__(self, obstacles, bounds=[[0, 0], [0, 1], [1, 0], [1, 1]]):
        self.obstacles = obstacles
        
        super().__init__(bounds + points_from_obstacles(self.obstacles))
        
        self.centroids = np.array([
            get_centroid(self.points[simplex, :]) for simplex in self.simplices
        ])

        self.passable, self.impassable = self._partition_simplices()
        self.dual_graph = self._create_dual_graph()

    def _partition_simplices(self):
        impassable = set()
        passable = set()

        for i, centroid in enumerate(self.centroids):
            inside = False
            for obstacle in self.obstacles:
                if obstacle.is_inside(centroid):
                    inside = True
                    break
            
            if inside:
                impassable.add(i)
            else:
                passable.add(i)
        
        return passable, impassable

    def _create_dual_graph(self):
        n = len(self.passable)
        graph_mat = np.zeros((n, n))
        for i, simplex in enumerate(self.simplices):
            if i in self.impassable: continue

            neighbors = self.neighbors[i]
            centroid_a = self.centroids[i]

            for j in neighbors:
                if (j != -1) and j not in self.impassable:
                    centroid_b = self.centroids[j]
                    dist = np.linalg.norm(centroid_a - centroid_b)
                    graph_mat[i, j] = dist

        return csr_matrix(graph_mat)
    
    def find_path(self, start_point, end_point):
        start_simplex, end_simplex = self.find_simplex([start_point, end_point])
        
        if start_simplex == -1 or start_simplex in self.impassable:
            raise Exception("starting point is inaccessible")
        
        if end_simplex == -1 or end_simplex in self.impassable:
            raise Exception("ending point is inaccessible")

        _, predecessors = dijkstra(csgraph=self.dual_graph, indices=start_simplex, directed=False, return_predecessors=True)

        path = [end_simplex]
        current_index = end_simplex

        while current_index != start_simplex:
            previous = predecessors[current_index]
            if previous >=0:
                path.insert(0, previous)
                current_index = previous
            else:
                raise Exception('no path exists')

        return path

    def plot(self):
        impassable_simplices = self.simplices[list(self.impassable), :]
        passable_simplices = self.simplices[list(self.passable), :]
        plt.triplot(self.points[:, 0], self.points[:, 1], impassable_simplices, color='r')
        plt.triplot(self.points[:, 0], self.points[:, 1], passable_simplices, color='g')
        plt.plot(self.points[:, 0], self.points[:, 1], 'o', color='b')


    def plot_path(self, path, color='black'):
        points = self.centroids[path, :]

        plt.plot(points[:, 0], points[:, 1], '-', color=color)


if __name__ == "__main__":
    from obstacles import Circle
    from matplotlib import pyplot as plt


    obstacles = [
        Circle([0.5, 0.5], 0.25)
    ]

    nav = NavMesh(obstacles)
    path = nav.find_path([0.1, 0.1], [0.9, 0.9])

    nav.plot()
    nav.plot_path(path)

    plt.gca().set_aspect('equal')
    plt.savefig('./images/nav.png')