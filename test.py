import matplotlib.pyplot as plt
import pyximport

import numpy as np
import shapely

pyximport.install(setup_args={'include_dirs': np.get_include()})

from pathfinder import Pathfinder

from obstacles import Circle

import timeit


def two_circles():
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
    pf.plot(plt, lines_of_sight=False)
    plt.gca().set_aspect('equal')
    plt.show()


# print(timeit.timeit(lambda: two_circles(), number=5) / 5)
two_circles()