"""A demo of the probabilistic pathfinder built for AUVSI-SUAS

Author: Drew Wagner (UAV Concordia)
"""
from .search_space import BoxSearchSpace
from .obstacles import Cylinder

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np

import time

print(np.random.seed(43))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

box = BoxSearchSpace()

while len(box.obstacles) < 10:
    p = np.random.uniform(
        low=0.2, high=0.8, size=(3,))
    p[2] = 0
    cyl = Cylinder(
        p,
        np.random.uniform(low=0.01, high=0.15),
        np.random.uniform(low=0.05))

    if not list(box.obstacle_index.intersection(cyl.bounds)):
        box.insert_obstacle(cyl)

t = time.time()
box.construct(n=100, k=6)
print("Time to construct", time.time() - t)

box.plot(ax, obstacles=True, graph=False, vertices=False)

a = np.array([0, 0, 0.25])
b = np.array([1, 1, 0.5])

t = time.time()
path = box.find_path(a, b, plot=ax)

print("Time to find path", time.time() - t)
plt.show()
