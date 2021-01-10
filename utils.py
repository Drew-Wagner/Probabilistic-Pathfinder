import numpy as np

### Segment intersection solution found here: https://stackoverflow.com/a/9997374/14080900

def _ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    # "Intersections" at endpoints should not count
    if np.isclose(A, C).all() or np.isclose(A, D).all() or np.isclose(B, C).all() or np.isclose(B, D).all():
        return False

    return _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D)

