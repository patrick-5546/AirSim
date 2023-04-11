import numpy as np

import pnt2line


def main():
    # print_evaluations()

    z = -10
    pts = [
        np.array([x, y, z])
        for x, y in [
            (0, -1), (128, -1), (128, 127), (0, 127),
            (0, -1), (128, -1), (128, -128), (0, -128),
            (0, -1),
        ]
    ]
    bot_pts = [
        np.array([1.04, -1.84, -12.14]),
        np.array([195.73, -1.82, -10.65]),
        np.array([0, -2, -10]),
    ]
    bot_pt = bot_pts[1]
    print(f'{dist_orig(pts, 0, bot_pt):.2f}', f'{dist_new(pts, 0, bot_pt):.2f}', f'{dist_pts(pts, 0, bot_pt):.2f}')


def print_evaluations(filename='evaluations.npz'):
    evaluations = np.load(filename)
    evaluations.files
    for item in evaluations.files:
        print(item, evaluations[item], sep='\n', end='\n\n')


def dist_orig(pts, i, bot_pt):
    return np.linalg.norm(np.cross((bot_pt - pts[i]), (bot_pt - pts[i + 1]))) / np.linalg.norm(pts[i] - pts[i + 1])


def dist_new(pts, i, bot_pt):
    return pnt2line.pnt2line(bot_pt, pts[i], pts[i + 1])[0]


def dist_pts(pts, i, bot_pt):
    return min(np.linalg.norm(bot_pt - pts[i]), np.linalg.norm(bot_pt - pts[i + 1]))


if __name__ == '__main__':
    main()
