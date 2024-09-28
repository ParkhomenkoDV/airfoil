from numpy import array, zeros, linspace
from scipy.interpolate import BPoly
import matplotlib.pyplot as plt

from decorators import timeit


@timeit()
def bezier_curve(points, N: int = 10):
    """Кривая Безье"""

    def bezier_value(d1: float | int, d2: float | int, t: float | int) -> float:
        """Вес Безье"""
        return d1 + (d2 - d1) * t

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
                x0, y0 = bezier_value(xt[0], xt[1], i / (N - 1)), bezier_value(yt[0], yt[1], i / (N - 1))
                x1, y1 = bezier_value(xt[1], xt[2], i / (N - 1)), bezier_value(yt[1], yt[2], i / (N - 1))
                p[i][0], p[i][1] = bezier_value(x0, x1, i / (N - 1)), bezier_value(y0, y1, i / (N - 1))
                break
            else:
                xN, yN = [], []
                for j in range(len(xt) - 1):
                    xN.append(bezier_value(xt[j], xt[j + 1], i / (N - 1)))
                    yN.append(bezier_value(yt[j], yt[j + 1], i / (N - 1)))
                xt, yt = xN, yN
    return p


# 0.0002 seconds
def bernstein_curve(points, N: int = 10):
    """Кривая Бернштейна"""
    points = array(points)[:, None, :]  # добавление новой оси в массиве
    curve = BPoly(points, [0, 1])
    t = linspace(0, 1, N)
    return curve(t)


def show(*args, title='curve'):
    plt.title(title, fontsize=14)
    plt.grid(True)  # сетка
    for points in args: plt.plot(*points.T, ls='solid')
    plt.axis('equal')
    plt.show()


def test() -> None:
    """Тестирование"""
    points = ((1, 0), (0.4, 0.4), (0.05, 0.15), (0, 0), (0.1, -0.1), (0.2, -0.1), (0.5, 0.15), (1, 0))
    points = array(points)

    bezier_points = bezier_curve(points, N=1_000)
    # print(bezier_points)
    show(points, bezier_points, title='Bezier curve')

    bernstein_points = bernstein_curve(points, N=1_000)
    # print(bernstein_points)
    show(points, bernstein_points, title='Bernstein curve')


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
