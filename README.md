# Probabilistic Pathfinder
A 3D probabilistic pathfinder built for the 2021 AUVSI-SUAS UAV competition.

## Obstacles
Currently only box search spaces and cylinderical obstacles are supported.

Obstacles are represented by triangular meshes. This allows efficient line segment intersections
using the [Moller-Trumbore ray-triangle intersection algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)

Additionally, obstacles must have a well-defined inside.
