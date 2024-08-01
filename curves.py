import sys
from numpy import array, zeros, linspace, sqrt, pi, linalg, tan
from scipy.interpolate import BPoly
import matplotlib.pyplot as plt

sys.path.append('D:/Programming/Python/scripts')

from decorators import timeit


def BezierValue(d1: float, d2: float, t: float) -> float:
    """Вес Безье"""
    return d1 + (d2 - d1) * t


@timeit()
def BezierCurve(points, N: int = 10):
    """Кривая Безье"""
    '''if type(points) is not array:
        print('points is not list');
        return
        if type(xp) is list and type(yp) is list and len(xp) != len(yp): print('len(x)=/=len(y)'); return
        if type(N) is not int or (type(N) is int and (N <= 1)): print('N is not int >= 2'); return
        if type(show) is not bool:
            print('show is not bool');
            return
    else:'''
    points = array(points)
    p = zeros((N, 2))
    for i in range(N):
        xt, yt = points[:, 0], points[:, 1]
        while True:
            if len(xt) == 3:
                x0, y0 = BezierValue(xt[0], xt[1], i / (N - 1)), BezierValue(yt[0], yt[1], i / (N - 1))
                x1, y1 = BezierValue(xt[1], xt[2], i / (N - 1)), BezierValue(yt[1], yt[2], i / (N - 1))
                p[i][0], p[i][1] = BezierValue(x0, x1, i / (N - 1)), BezierValue(y0, y1, i / (N - 1))
                break
            else:
                xN, yN = [], []
                for j in range(len(xt) - 1):
                    xN.append(BezierValue(xt[j], xt[j + 1], i / (N - 1)))
                    yN.append(BezierValue(yt[j], yt[j + 1], i / (N - 1)))
                xt, yt = xN, yN
    return p


# 0.0002 seconds
def BernsteinCurve(points, N: int = 10):
    """Кривая Бернштейна"""
    points = array(points)[:, None, :]  # добавление новой оси в массиве
    curve = BPoly(points, [0, 1])
    t = linspace(0, 1, N)
    p = curve(t)
    return p


def show(*args, title='curve'):
    plt.title(title, fontsize=14)
    plt.grid(True)  # сетка
    for points in args: plt.plot(*points.T, '-')
    plt.show()


def parsec_coef(xte, yte, rle, x_cre, y_cre, d2ydx2_cre, th_cre, surface):
    """PARSEC coefficients"""

    # Initialize coefficients
    coef = zeros(6)

    # 1st coefficient depends on surface (pressure or suction)
    coef[0] = -sqrt(2 * rle) if surface.startswith('p') else sqrt(2 * rle)

    # Form system of equations
    A = array([
        [xte ** 1.5, xte ** 2.5, xte ** 3.5, xte ** 4.5, xte ** 5.5],
        [x_cre ** 1.5, x_cre ** 2.5, x_cre ** 3.5, x_cre ** 4.5,
         x_cre ** 5.5],
        [1.5 * sqrt(xte), 2.5 * xte ** 1.5, 3.5 * xte ** 2.5,
         4.5 * xte ** 3.5, 5.5 * xte ** 4.5],
        [1.5 * sqrt(x_cre), 2.5 * x_cre ** 1.5, 3.5 * x_cre ** 2.5,
         4.5 * x_cre ** 3.5, 5.5 * x_cre ** 4.5],
        [0.75 * (1 / sqrt(x_cre)), 3.75 * sqrt(x_cre), 8.75 * x_cre ** 1.5,
         15.75 * x_cre ** 2.5, 24.75 * x_cre ** 3.5]
    ])

    B = array([
        [yte - coef[0] * sqrt(xte)],
        [y_cre - coef[0] * sqrt(x_cre)],
        [tan(th_cre * pi / 180) - 0.5 * coef[0] * (1 / sqrt(xte))],
        [-0.5 * coef[0] * (1 / sqrt(x_cre))],
        [d2ydx2_cre + 0.25 * coef[0] * x_cre ** (-1.5)]
    ])

    # Solve system of linear equations
    X = linalg.solve(A, B)

    # Gather all coefficients
    coef[1:6] = X[0:5, 0]

    # Return coefficients
    return coef


if __name__ == '__main__':
    points = [(0, 0), (0.05, 0.15), (0.4, 0.4), (1, 0)]
    points = array(points)

    bezier_points = BezierCurve(points, N=1000)
    # print(bezier_points)
    show(points, bezier_points, title='Bezier curve')

    bernstein_points = BernsteinCurve(points, N=1000)
    # print(bernstein_points)
    show(points, bernstein_points, title='Bernstein curve')
