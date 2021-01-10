import numpy as np


T = np.array([[0, -1], [1, 0]])


def line_intersect(a1, a2, b1, b2):
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1


def segments_intersect(a1, a2, b1, b2):
    line_intersection = line_intersect(a1, a2, b1, b2).ravel()
    if not np.isfinite(line_intersection).all():
        return False

    seg_a = a2 - a1
    seg_b = b2 - b1
    seg_a_len = np.linalg.norm(seg_a)
    seg_b_len = np.linalg.norm(seg_b)

    seg_c = line_intersection - a1
    seg_d = line_intersection - b1
    cross_a = np.cross(seg_a, seg_c)
    cross_b = np.cross(seg_b, seg_d)

    if cross_a != 0 or cross_b != 0:
        return False
    else:
        proj_a = np.dot(seg_c, seg_a)
        proj_b = np.dot(seg_d, seg_b)
        return 0 <= proj_a <= seg_a_len and 0 <= proj_b <= seg_b_len


if __name__ == "__main__":
    assert segments_intersect(
        np.array([-1, 0]),
        np.array([1, 0]),
        np.array([0, 0]),
        np.array([0, 1]))

    assert segments_intersect(
        np.array([-1, 0]),
        np.array([1, 0]),
        np.array([-1, 0]),
        np.array([1, 1]))

    assert segments_intersect(
        np.array([-1, 0]),
        np.array([1, 0]),
        np.array([0, -1]),
        np.array([0, 1]))

    assert not segments_intersect(
        np.array([-1, 0]),
        np.array([1, 0]),
        np.array([0, 0.5]),
        np.array([0, 1]))
