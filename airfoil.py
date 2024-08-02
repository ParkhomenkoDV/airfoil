import sys
from tqdm import tqdm
from colorama import Fore

from math import radians, degrees, sqrt, floor, ceil
import numpy as np
from numpy import linspace, arange, array, nan, inf, pi, cos, sin, tan, arctan
from scipy import interpolate, integrate
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from curves import bernstein_curve

sys.path.append('D:/Programming/Python/scripts')

from tools import export2, isnum, COOR, LINE, Axis, angle, rounding, eps, dist, dist2line, isiter
from decorators import timeit

import cProfile


class Airfoil:
    """Относительный аэродинамический профиль"""

    rnd = 4  # количество значащих цифр
    __Nrecomend = 30  # рекомендуемое количество дискретных точек

    def __init__(self, method: str, N: int = __Nrecomend):
        self.__method = method  # метод построения аэродинамического профиля
        self.__N = N  # количество точек дискретизации
        self.coords = {'u': {'x': list(), 'y': list()},
                       'l': {'x': list(), 'y': list()}}
        self.__props = dict()

    def __str__(self) -> str:
        return self.__method

    @property
    def method(self) -> str:
        return self.__method

    @method.setter
    def method(self, method) -> None:
        self.reset()
        if type(method) is str and method.strip():
            self.__method = method.strip()
        else:
            print(Fore.RED + f'method is str in []!' + Fore.RESET)

    @method.deleter
    def method(self) -> None:
        self.reset()

    @property
    def N(self) -> int:
        return self.__N

    @N.setter
    def N(self, N) -> None:
        self.reset()
        if type(N) is int and N >= 3:
            self.__N = N
        else:
            print(Fore.RED + f'N is int >= 3!' + Fore.RESET)

    @N.deleter
    def N(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.__method = ''
        self.__N = self.__Nrecomend
        self.coords = {'u': {'x': [], 'y': []},
                       'l': {'x': [], 'y': []}}
        self.__props = dict()

    def input(self):
        pass

    def validate(self) -> bool:
        assert_sms = 'Incorrect condition: '
        if self.method.upper() in ('BMSTU', 'МГТУ', 'МВТУ', 'МИХАЛЬЦЕВ'):
            # относ. координата пересечения входного и выходного лучей
            assert hasattr(self, 'xg_b'), assert_sms + 'hasattr(self, "xg_b")'
            assert type(self.xg_b) in (int, float), assert_sms + 'type(self.xg_b) in (int, float)'
            assert 0 <= self.xg_b <= 1, assert_sms + '0 <= self.xg_b <= 1'

            # относ. радиус входной кромоки
            assert hasattr(self, 'r_inlet_b'), assert_sms + 'hasattr(self, "r_inlet_b")'
            assert type(self.r_inlet_b) in (int, float), assert_sms + 'type(self.r_inlet_b) in (int, float)'
            assert 0 <= self.r_inlet_b <= 1, assert_sms + '0 <= self.r_inlet_b <= 1'

            # относ. радиус выходной кромки
            assert hasattr(self, 'r_outlet_b'), assert_sms + 'hasattr(self, "r_outlet_b")'
            assert type(self.r_outlet_b) in (int, float), assert_sms + 'type(self.r_outlet_b) in (int, float)'
            assert 0 <= self.r_outlet_b <= 1, assert_sms + '0 <= self.r_outlet_b <= 1'

            # степень приближенности к спинке
            assert hasattr(self, 'g_'), assert_sms + 'hasattr(self, "g_")'
            assert type(self.g_) in (int, float), assert_sms + 'type(self.g_) in  (int,float)'
            assert 0 <= self.g_ <= 1, assert_sms + '0 <= self.g_ <= 1'

            # угол раскрытия входной кромоки
            assert hasattr(self, 'g_inlet'), assert_sms + 'hasattr(self, "g_inlet")'
            assert type(self.g_inlet) in (int, float), assert_sms + 'type(self.g_inlet) in (int,float)'
            assert 0 <= self.g_inlet, assert_sms + '0 <= self.g_inlet'

            # угол раскрытия выходной кромки
            assert hasattr(self, 'g_outlet'), assert_sms + 'hasattr(self, "g_outlet")'
            assert type(self.g_outlet) in (int, float), assert_sms + 'type(self.g_outlet) in (int, float)'
            assert 0 <= self.g_outlet, assert_sms + '0 <= self.g_outlet'

            # угол поворота потока
            assert hasattr(self, 'e'), assert_sms + 'hasattr(self, "e")'
            assert type(self.e) in (int, float), assert_sms + 'type(self.e) in (int, float)'

        if self.method.upper() in ('NACA', 'N.A.C.A.'):
            # относ. максимальная толщина профиля
            assert hasattr(self, 'c_b'), assert_sms + 'hasattr(self, "c_b")'
            assert type(self.c_b) in (int, float), assert_sms + 'type(self.c_b) in (int, float)'
            assert 0 <= self.c_b <= 1, assert_sms + '0 <= self.c_b <= 1'

            # относ. координата максимального прогиба профиля
            assert hasattr(self, 'xf_b'), assert_sms + 'hasattr(self, "xf_b")'
            assert type(self.xf_b) in (int, float), assert_sms + 'type(self.xf_b) in (int, float)'
            assert 0 <= self.xf_b <= 1, assert_sms + '0 <= self.xf_b <= 1'

            # относ. максимальный прогиб профиля
            assert hasattr(self, 'f_b'), assert_sms + 'hasattr(self, "f_b")'
            assert type(self.f_b) in (int, float), assert_sms + 'type(self.f_b) in (int, float)'
            assert 0 <= self.f_b <= 1, assert_sms + '0 <= self.f_b <= 1'

        if self.method.upper() in ('MYNK', 'МУНК'):
            assert hasattr(self, 'h'), assert_sms + 'hasattr(self, "h")'
            assert type(self.h) in (int, float), assert_sms + 'type(self.h) in (int, float)'
            assert 0 <= self.h <= 1, assert_sms + '0 <= self.h <= 1'

        if self.method.upper() in ('PARSEC',):
            assert hasattr(self, 'r_inlet_b'), assert_sms + 'hasattr(self, "r_inlet_b")'
            assert type(self.r_inlet_b) in (int, float), assert_sms + 'type(self.r_inlet_b) in (int, float)'
            assert 0 <= self.r_inlet_b <= 1, assert_sms + '0 <= self.r_inlet_b <= 1'

        if self.method.upper() in ('BEZIER', 'БЕЗЬЕ'):
            assert hasattr(self, 'u'), assert_sms + 'hasattr(self, "u")'
            assert isiter(self.u), assert_sms + 'isiter(self.u)'
            assert all(map(isiter, self.u)), assert_sms + 'all(map(isiter, self.u))'
            assert all(len(el) == 2 for el in self.u), assert_sms + 'all(map(isiter, self.u))'
            assert all(type(el) in (int, float) for itr in self.u for el in itr), (
                    assert_sms + 'all(type(el) in (int, float) for itr in self.u for el in itr)')

            assert hasattr(self, 'l'), assert_sms + 'hasattr(self, "l")'
            assert isiter(self.l), assert_sms + 'isiter(self.l)'
            assert all(map(isiter, self.l)), assert_sms + 'all(map(isiter, self.l))'
            assert all(len(el) == 2 for el in self.l), assert_sms + 'all(map(isiter, self.l))'
            assert all(type(el) in (int, float) for itr in self.l for el in itr), (
                    assert_sms + 'all(type(el) in (int, float) for itr in self.l for el in itr)')

        if self.method.upper() in ('MANUAL', 'ВРУЧНУЮ'):
            pass

        return True

    def BMSTU(self):
        # tan угла входа и выхода потока
        k_inlet, k_outlet = 1 / (2 * self.xg_b / (self.xg_b - 1) * tan(self.e)), 1 / (2 * tan(self.e))
        if tan(self.e) * self.e > 0:
            k_inlet *= ((self.xg_b / (self.xg_b - 1) - 1) -
                        sqrt((self.xg_b / (self.xg_b - 1) - 1) ** 2 -
                             4 * (self.xg_b / (self.xg_b - 1) * tan(self.e) ** 2)))
            k_outlet *= ((self.xg_b / (self.xg_b - 1) - 1) -
                         sqrt((self.xg_b / (self.xg_b - 1) - 1) ** 2 -
                              4 * (self.xg_b / (self.xg_b - 1) * tan(self.e) ** 2)))
        else:
            k_inlet *= ((self.xg_b / (self.xg_b - 1) - 1) +
                        sqrt((self.xg_b / (self.xg_b - 1) - 1) ** 2 -
                             4 * (self.xg_b / (self.xg_b - 1) * tan(self.e) ** 2)))
            k_outlet *= ((self.xg_b / (self.xg_b - 1) - 1) +
                         sqrt((self.xg_b / (self.xg_b - 1) - 1) ** 2 -
                              4 * (self.xg_b / (self.xg_b - 1) * tan(self.e) ** 2)))

        # углы входа и выхода профиля
        if self.e > 0:
            g_u_inlet, g_d_inlet = (1 - self.g_) * self.g_inlet, self.g_ * self.g_inlet
            g_u_outlet, g_d_outlet = (1 - self.g_) * self.g_outlet, self.g_ * self.g_outlet
        else:
            g_u_inlet, g_d_inlet = self.g_ * self.g_inlet, (1 - self.g_) * self.g_inlet,
            g_u_outlet, g_d_outlet = self.g_ * self.g_outlet, (1 - self.g_) * self.g_outlet

        # положения центров окружностей входной и выходной кромок
        self.__O_inlet = self.r_inlet_b, k_inlet * self.r_inlet_b
        self.__O_outlet = 1 - self.r_outlet_b, -k_outlet * self.r_outlet_b

        # точки пересечения линий спинки и корыта
        xcl_u, ycl_u = COOR(tan(arctan(k_inlet) + g_u_inlet),
                            sqrt(tan(arctan(k_inlet) + g_u_inlet) ** 2 + 1) * self.r_inlet_b -
                            (tan(arctan(k_inlet) + g_u_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1],
                            tan(arctan(k_outlet) - g_u_outlet),
                            sqrt(tan(arctan(k_outlet) - g_u_outlet) ** 2 + 1) * self.r_outlet_b -
                            (tan(arctan(k_outlet) - g_u_outlet)) * self.__O_outlet[0] - (-1) * self.__O_outlet[1])

        xcl_d, ycl_d = COOR(tan(arctan(k_inlet) - g_d_inlet),
                            -sqrt(tan(arctan(k_inlet) - g_d_inlet) ** 2 + 1) * self.r_inlet_b -
                            (tan(arctan(k_inlet) - g_d_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1],
                            tan(arctan(k_outlet) + g_d_outlet),
                            -sqrt(tan(arctan(k_outlet) + g_d_outlet) ** 2 + 1) * self.r_outlet_b -
                            (tan(arctan(k_outlet) + g_d_outlet)) * self.__O_outlet[0] - (-1) * self.__O_outlet[1])

        # точки пересечения окружностей со спинкой и корытом
        xclc_i_u, yclc_i_u = COOR(tan(arctan(k_inlet) + g_u_inlet),
                                  sqrt(tan(arctan(k_inlet) + g_u_inlet) ** 2 + 1) * self.r_inlet_b
                                  - (tan(arctan(k_inlet) + g_u_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1],
                                  -1 / (tan(arctan(k_inlet) + g_u_inlet)),
                                  -(-1 / tan(arctan(k_inlet) + g_u_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[
                                      1])

        xclc_i_d, yclc_i_d = COOR(tan(arctan(k_inlet) - g_d_inlet),
                                  -sqrt(tan(arctan(k_inlet) - g_d_inlet) ** 2 + 1) * self.r_inlet_b
                                  - (tan(arctan(k_inlet) - g_d_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1],
                                  -1 / (tan(arctan(k_inlet) - g_d_inlet)),
                                  -(-1 / tan(arctan(k_inlet) - g_d_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[
                                      1])

        xclc_e_u, yclc_e_u = COOR(tan(arctan(k_outlet) - g_u_outlet),
                                  sqrt(tan(arctan(k_outlet) - g_u_outlet) ** 2 + 1) * self.r_outlet_b
                                  - tan(arctan(k_outlet) - g_u_outlet) * self.__O_outlet[0] - (-1) * self.__O_outlet[1],
                                  -1 / tan(arctan(k_outlet) - g_u_outlet),
                                  -(-1 / tan(arctan(k_outlet) - g_u_outlet)) * self.__O_outlet[0] - (-1) *
                                  self.__O_outlet[1])

        xclc_e_d, yclc_e_d = COOR(tan(arctan(k_outlet) + g_d_outlet),
                                  -sqrt(tan(arctan(k_outlet) + g_d_outlet) ** 2 + 1) * self.r_outlet_b
                                  - tan(arctan(k_outlet) + g_d_outlet) * self.__O_outlet[0] - (-1) * self.__O_outlet[1],
                                  -1 / tan(arctan(k_outlet) + g_d_outlet),
                                  -(-1 / tan(arctan(k_outlet) + g_d_outlet)) * self.__O_outlet[0] - (-1) *
                                  self.__O_outlet[1])

        # точки входной окружности кромки по спинке
        an = angle(points=((0, self.__O_inlet[1]), self.__O_inlet, (xclc_i_u, yclc_i_u)))
        if xclc_i_u > self.__O_inlet[0]: an = pi - an
        for a in arange(0, an, an / self.__N):
            self.coords['u']['x'].append(self.r_inlet_b * (1 - cos(a)))
            self.coords['u']['y'].append(self.__O_inlet[1] + self.r_inlet_b * sin(a))

        # точки входной окружности кромки по корыту
        an = angle(points=((0, self.__O_inlet[1]), self.__O_inlet, (xclc_i_d, yclc_i_d)))
        if xclc_i_d > self.__O_inlet[0]: an = pi - an
        for a in arange(0, an, an / self.__N):
            self.coords['l']['x'].append(self.r_inlet_b * (1 - cos(a)))
            self.coords['l']['y'].append(self.__O_inlet[1] - self.r_inlet_b * sin(a))

        xu, yu = bernstein_curve(((xclc_i_u, yclc_i_u), (xcl_u, ycl_u), (xclc_e_u, yclc_e_u)), N=self.__N).T.tolist()
        xd, yd = bernstein_curve(((xclc_i_d, yclc_i_d), (xcl_d, ycl_d), (xclc_e_d, yclc_e_d)), N=self.__N).T.tolist()
        self.coords['u']['x'] += xu
        self.coords['u']['y'] += yu
        self.coords['l']['x'] += xd
        self.coords['l']['y'] += yd

        an = angle(points=((1, self.__O_outlet[1]), self.__O_outlet, (xclc_e_u, yclc_e_u)))
        if self.__O_outlet[0] > xclc_e_u: an = pi - an
        for a in arange(0, an, an / self.__N):
            self.coords['u']['x'].insert(2 * self.__N, 1 - self.r_outlet_b * (1 - cos(a)))
            self.coords['u']['y'].insert(2 * self.__N, self.__O_outlet[1] + self.r_outlet_b * sin(a))

        an = angle(points=((1, self.__O_outlet[1]), self.__O_outlet, (xclc_e_d, yclc_e_d)))
        if self.__O_outlet[0] > xclc_e_d: an = pi - an
        for a in arange(0, an, an / self.__N):
            self.coords['l']['x'].insert(2 * self.__N, 1 - self.r_outlet_b * (1 - cos(a)))
            self.coords['l']['y'].insert(2 * self.__N, self.__O_outlet[1] - self.r_outlet_b * sin(a))

    def NACA(self, closed=True):
        i = arange(self.__N)
        betta = i * pi / (2 * (self.__N - 1))
        x = 1 - np.cos(betta)

        mask = (0 <= x) & (x <= self.xf_b)

        yf = np.full_like(i, self.f_b, dtype=np.float64)
        yf[mask] *= self.xf_b ** (-2) * (2 * self.xf_b * x[mask] - x[mask] ** 2)
        yf[~mask] *= (1 - self.xf_b) ** (-2) * (1 - 2 * self.xf_b + 2 * self.xf_b * x[~mask] - x[~mask] ** 2)

        gradYf = 2 * self.f_b

        a = np.array([0.2969, -0.126, -0.3516, 0.2843, -0.1036 if closed else -0.1015])

        yc = self.c_b / 0.2 * np.dot(a, np.column_stack((np.sqrt(x), x, x ** 2, x ** 3, x ** 4)).T)

        tetta = np.arctan(gradYf)

        sin_tetta, cos_tetta = np.sin(tetta), np.cos(tetta)  # предварительный расчет для ускорения работы

        self.coords['u']['x'], self.coords['u']['y'] = x - yc * sin_tetta, yf + yc * cos_tetta
        self.coords['l']['x'], self.coords['l']['y'] = x + yc * sin_tetta, yf - yc * cos_tetta

        x_min = min(np.min(self.coords['u']['x']), np.min(self.coords['l']['x']))
        x_max = max(np.max(self.coords['u']['x']), np.max(self.coords['l']['x']))
        scale = abs(x_max - x_min)
        self.transform(x0=x_min, scale=1 / scale, inplace=True)

        # отсечка значений спинки корыту и наоборот
        Xu = (self.coords['u']['x'][self.coords['u']['x'].index(min(self.coords['u']['x'])):] +
              list(reversed(self.coords['l']['x'][
                            self.coords['l']['x'].index(max(self.coords['l']['x'])):len(self.coords['l']['x']) - 1])))
        Yu = (self.coords['u']['y'][self.coords['u']['x'].index(min(self.coords['u']['x'])):] +
              list(reversed(self.coords['l']['y'][
                            self.coords['l']['x'].index(max(self.coords['l']['x'])):len(self.coords['l']['x']) - 1])))
        Xd = list(reversed(self.coords['u']['x'][1:self.coords['u']['x'].index(min(self.coords['u']['x'])) + 1])) + \
             self.coords['l']['x'][:self.coords['l']['x'].index(max(self.coords['l']['x'])) + 1]
        Yd = list(reversed(self.coords['u']['y'][1:self.coords['u']['x'].index(min(self.coords['u']['x'])) + 1])) + \
             self.coords['l']['y'][:self.coords['l']['x'].index(max(self.coords['l']['x'])) + 1]

        self.coords['u']['x'], self.coords['u']['y'] = Xu, Yu
        self.coords['l']['x'], self.coords['l']['y'] = Xd, Yd

        self.find_circles()

    def MYNK(self):
        x = linspace(0, 1, self.__N)
        self.coords['u']['x'] = x
        self.coords['u']['y'] = self.h * (0.25 * (-x - 17 * x ** 2 - 6 * x ** 3) + x ** 0.87 * (1 - x) ** 0.56)
        self.coords['l']['x'] = x
        self.coords['l']['y'] = self.h * (0.25 * (-x - 17 * x ** 2 - 6 * x ** 3) - x ** 0.87 * (1 - x) ** 0.56)

        angle = arctan((self.coords['u']['y'][-1] - self.coords['u']['y'][0]) / (1 - 0))
        scale = dist((self.coords['u']['x'][0], self.coords['u']['y'][0]),
                     (self.coords['u']['x'][-1], self.coords['u']['y'][-1]))
        self.transform(angle=angle, scale=1 / scale, inplace=True)

        self.find_circles()

    def __parsec_coefficients(self, surface: str,
                              radius_inlet: float | int,
                              c_b: tuple, d2y_dx2_surface,
                              outlet: tuple, thetta_outlet_surface: float | int):
        """PARSEC coefficients"""
        assert surface in ('l', 'u')

        x_c_b, y_c_b = c_b
        x_outlet, y_outlet = outlet

        coef = np.zeros(6)  # Initialize coefficients

        # 1st coefficient depends on surface (pressure or suction)
        coef[0] = -sqrt(2 * radius_inlet) if surface == 'l' else sqrt(2 * radius_inlet)

        # Form system of equations
        A = array([
            [x_outlet ** 1.5, x_outlet ** 2.5, x_outlet ** 3.5, x_outlet ** 4.5, x_outlet ** 5.5],
            [x_c_b ** 1.5, x_c_b ** 2.5, x_c_b ** 3.5, x_c_b ** 4.5, x_c_b ** 5.5],
            [1.5 * sqrt(x_outlet), 2.5 * x_outlet ** 1.5, 3.5 * x_outlet ** 2.5, 4.5 * x_outlet ** 3.5,
             5.5 * x_outlet ** 4.5],
            [1.5 * sqrt(x_c_b), 2.5 * x_c_b ** 1.5, 3.5 * x_c_b ** 2.5, 4.5 * x_c_b ** 3.5, 5.5 * x_c_b ** 4.5],
            [0.75 * (1 / sqrt(x_c_b)), 3.75 * sqrt(x_c_b), 8.75 * x_c_b ** 1.5, 15.75 * x_c_b ** 2.5,
             24.75 * x_c_b ** 3.5]
        ])

        B = array([
            [y_outlet - coef[0] * sqrt(x_outlet)],
            [y_c_b - coef[0] * sqrt(x_c_b)],
            [tan(thetta_outlet_surface * pi / 180) - 0.5 * coef[0] * (1 / sqrt(x_outlet))],
            [-0.5 * coef[0] * (1 / sqrt(x_c_b))],
            [d2y_dx2_surface + 0.25 * coef[0] * x_c_b ** (-1.5)]
        ])

        X = np.linalg.solve(A, B)  # Solve system of linear equations
        coef[1:6] = X[0:5, 0]  # Gather all coefficients

        return coef

    def PARSEC(self):
        """
        Generate and plot the contour of an airfoil using the PARSEC parameterization
        H. Sobieczky, *'Parametric airfoils and wings'* in *Notes on Numerical Fluid Mechanics*, Vol. 68, pp 71-88]
        (www.as.dlr.de/hs/h-pdf/H141.pdf)
        Repository & documentation: http://github.com/dqsis/parsec-airfoils
        """

        # поверхностные коэффициенты давления спинки и корыта
        cf_u = self.__parsec_coefficients('u', self.r_inlet_b,
                                          (self.x_suc, self.y_suc), self.d2y_dx2_u,
                                          (1, 0), self.th_suc)
        cf_l = self.__parsec_coefficients('l', self.r_inlet_b,
                                          (self.x_pre, self.y_pre), self.d2y_dx2_l,
                                          (1, 0), self.thetta_pre)

        self.coords['u']['x'] = linspace(0, 1, self.__N)
        self.coords['u']['y'] = sum([cf_u[i] * self.coords['u']['x'] ** (i + 0.5) for i in range(6)])
        self.coords['l']['x'] = linspace(1, 0, self.__N)
        self.coords['l']['y'] = sum([cf_l[i] * self.coords['l']['x'] ** (i + 0.5) for i in range(6)])
        self.coords['l']['x'], self.coords['l']['y'] = self.coords['l']['x'][::-1], self.coords['l']['y'][::-1]

        self.find_circles()

    def Bezier(self):
        self.coords['u']['x'], self.coords['u']['y'] = bernstein_curve(self.u, N=self.__N).T
        self.coords['l']['x'], self.coords['l']['y'] = bernstein_curve(self.l, N=self.__N).T

        self.find_circles()

    @timeit()
    def solve(self):
        self.__props = dict()
        self.coords['u'], self.coords['l'] = {'x': [], 'y': []}, {'x': [], 'y': []}
        self.validate()

        if self.method.upper() in ('NACA', 'N.A.C.A.'):
            self.NACA()
        elif self.method.upper() in ('BMSTU', 'МГТУ', 'МВТУ', 'МИХАЛЬЦЕВ'):
            self.BMSTU()
        elif self.method.upper() in ('MYNK', 'МУНК'):
            self.MYNK()
        elif self.method.upper() in ('PARSEC',):
            self.PARSEC()
        elif self.method.upper() in ('BEZIER', 'БЕЗЬЕ'):
            self.Bezier()
        elif self.method.upper() in ('MANUAL', 'ВРУЧНУЮ'):
            pass
        else:
            print(Fore.RED + f'No such method {self.method}!' + Fore.RESET)

    def transform(self, x0=0, y0=0, angle=0, scale=1, inplace=False) -> dict[str: dict]:
        """Перенос-поворот кривых спинки и корыта профиля"""

        xun, yun = [nan] * len(self.coords['u']['x']), [nan] * len(self.coords['u']['x'])
        xdn, ydn = [nan] * len(self.coords['l']['x']), [nan] * len(self.coords['l']['x'])

        for i in range(len(self.coords['u']['x'])):
            xun[i], yun[i] = Axis.transform(self.coords['u']['x'][i], self.coords['u']['y'][i],
                                            x0, y0, angle, scale)

        for i in range(len(self.coords['l']['x'])):
            xdn[i], ydn[i] = Axis.transform(self.coords['l']['x'][i], self.coords['l']['y'][i],
                                            x0, y0, angle, scale)

        if inplace:
            self.coords['u']['x'], self.coords['u']['y'] = xun, yun
            self.coords['l']['x'], self.coords['l']['y'] = xdn, ydn

        return {'u': {'x': xun, 'y': yun},
                'l': {'x': xdn, 'y': ydn}}

    def find_circles(self) -> dict:
        """Поиск радиусов окружностей входной и выходной кромок"""
        if all(hasattr(self, attr) for attr in ['_Airfoil__O_inlet', 'r_inlet_b', '_Airfoil__O_outlet', 'r_outlet_b']):
            return {'inlet': {'O': self.__O_inlet, 'r': self.r_inlet_b},
                    'outlet': {'O': self.__O_outlet, 'r': self.r_outlet_b}}

        Fu = interpolate.interp1d(*self.coords['u'].values(), kind='quadratic', fill_value='extrapolate')
        Fd = interpolate.interp1d(*self.coords['l'].values(), kind='quadratic', fill_value='extrapolate')

        dx = 0.000_1

        x0, x1 = 0, 1
        y0, y1 = Fu(x0), Fu(x1)

        x0u, x1u = dx, 1 - dx
        y0u, y1u = Fu(x0u), Fu(x1u)

        x0d, x1d = dx, 1 - dx
        y0d, y1d = Fd(x0d), Fd(x1d)

        A0u, B0u, C0u = LINE((x0, y0), (x0u, y0u)).values()
        A0d, B0d, C0d = LINE((x0, y0), (x0d, y0d)).values()
        A1u, B1u, C1u = LINE((x1, y1), (x1u, y1u)).values()
        A1d, B1d, C1d = LINE((x1, y1), (x1d, y1d)).values()

        # A для перпендикуляров
        AA0u, AA0d = -1 / A0u, -1 / A0d
        AA1u, AA1d = -1 / A1u, -1 / A1d

        # С для перпендикуляров
        CC0u, CC0d = y0u - AA0u * x0u, y0d - AA0d * x0d
        CC1u, CC1d = y1u - AA1u * x1u, y1d - AA1d * x1d

        # центры входной и выходной окружностей
        self.__O_inlet = COOR(AA0u, CC0u, AA0d, CC0d)
        self.__O_outlet = COOR(AA1u, CC1u, AA1d, CC1d)

        self.r_inlet_b = abs(self.__O_inlet[0] - x0)
        self.r_outlet_b = abs(self.__O_outlet[0] - x1)

        return {'inlet': {'O': self.__O_inlet, 'r': self.r_inlet_b},
                'outlet': {'O': self.__O_outlet, 'r': self.r_outlet_b}}

    def show(self, figsize=(14, 5.25), savefig=False):
        """Построение профиля"""
        fg = plt.figure(figsize=figsize)
        gs = fg.add_gridspec(1, 4)  # строки, столбцы

        fg.add_subplot(gs[0, 0])
        plt.title('Initial data')
        plt.grid(False)
        plt.axis('off')

        plt.plot([], label=f'method = {self.method}')
        plt.plot([], label=f'N = {self.N}')
        for key, value in self.__dict__.items():
            if '__' not in key and type(value) in (int, float):
                plt.plot([], label=f'{key} = {rounding(value, self.rnd)}')
        plt.legend(loc='upper center')

        fg.add_subplot(gs[0, 1])
        plt.title('Airfoil with structure')
        plt.grid(True)  # сетка
        plt.axis('equal')
        plt.xlim([0, 1])

        plt.plot(self.coords['u']['x'], self.coords['u']['y'], ls='-', color='blue', linewidth=2)
        plt.plot(self.coords['l']['x'], self.coords['l']['y'], ls='-', color='red', linewidth=2)
        x_inlet, y_inlet, x_outlet, y_outlet = [], [], [], []
        for alpha in linspace(0, 2 * pi, 360):
            x_inlet.append(self.r_inlet_b * cos(alpha) + self.__O_inlet[0])
            y_inlet.append(self.r_inlet_b * sin(alpha) + self.__O_inlet[1])
            x_outlet.append(self.r_outlet_b * cos(alpha) + self.__O_outlet[0])
            y_outlet.append(self.r_outlet_b * sin(alpha) + self.__O_outlet[1])
        plt.plot(x_inlet, y_inlet, ls='-', color='black', linewidth=1)
        plt.plot(x_outlet, y_outlet, ls='-', color='black', linewidth=1)

        fg.add_subplot(gs[0, 2])
        plt.title('Airfoil')
        plt.grid(True)  # сетка
        plt.axis('equal')
        plt.xlim([0, 1])

        plt.plot(self.coords['u']['x'], self.coords['u']['y'], ls='-', color='black', linewidth=2)
        plt.plot(self.coords['l']['x'], self.coords['l']['y'], ls='-', color='black', linewidth=2)

        fg.add_subplot(gs[0, 3])
        plt.title('Properties')
        plt.grid(False)
        plt.axis('off')

        for key, value in self.properties.items(): plt.plot([], label=f'{key} = {rounding(value, self.rnd)}')
        plt.legend(loc='upper center')

        if savefig:
            export2(plt, file_path='exports/airfoil', file_name='airfoil', file_extension='png', show_time=False)
        plt.tight_layout()
        plt.show()

    def to_dataframe(self, bears: str = 'pandas'):
        if bears.strip().lower() == 'pandas':
            return pd.DataFrame({'xu': pd.Series(airfoil.coords['u']['x']), 'yu': pd.Series(airfoil.coords['u']['y']),
                                 'xd': pd.Series(airfoil.coords['l']['x']), 'yd': pd.Series(airfoil.coords['l']['y'])})
        if bears.strip().lower() == 'polars':
            return pl.concat([pl.DataFrame({'xu': airfoil.coords['u']['x'], 'yu': airfoil.coords['u']['y']}),
                              pl.DataFrame({'xl': airfoil.coords['l']['x'], 'yl': airfoil.coords['l']['y']})],
                             how='horizontal')
        print(Fore.RED + 'Unknown bears!' + Fore.RESET)
        print('Use "pandas" or "polars"!')

    @property
    @timeit()
    def properties(self, epsrel: float = 1e-4) -> dict[str: float]:
        if self.__props: return self.__props

        Yup = interpolate.interp1d(*self.coords['u'].values(), kind='quadratic', fill_value='extrapolate')
        Ydown = interpolate.interp1d(*self.coords['l'].values(), kind='quadratic', fill_value='extrapolate')

        self.__props['a_b'] = integrate.dblquad(lambda _, __: 1, 0, 1, lambda xu: Ydown(xu), lambda xd: Yup(xd),
                                                epsrel=epsrel)[0]
        self.__props['xc_b'], self.__props['c_b'] = -1.0, 0
        self.__props['xf_b'], self.__props['f_b'] = -1.0, 0
        for x in linspace(0, 1, ceil(1 / epsrel)):
            if Yup(x) - Ydown(x) > self.__props['c_b']:
                self.__props['xc_b'], self.__props['c_b'] = x, Yup(x) - Ydown(x)
            if abs((Yup(x) + Ydown(x)) / 2) > abs(self.__props['f_b']):
                self.__props['xf_b'], self.__props['f_b'] = x, (Yup(x) + Ydown(x)) / 2
        self.__props['Sx'] = integrate.dblquad(lambda y, _: y, 0, 1, lambda xu: Ydown(xu), lambda xd: Yup(xd),
                                               epsrel=epsrel)[0]
        self.__props['Sy'] = integrate.dblquad(lambda _, x: x, 0, 1, lambda xu: Ydown(xu), lambda xd: Yup(xd),
                                               epsrel=epsrel)[0]
        self.__props['x0'] = self.__props['Sy'] / self.__props['a_b']
        self.__props['y0'] = self.__props['Sx'] / self.__props['a_b']
        self.__props['Jx'] = integrate.dblquad(lambda y, _: y ** 2, 0, 1, lambda xu: Ydown(xu), lambda xd: Yup(xd),
                                               epsrel=epsrel)[0]
        self.__props['Jy'] = integrate.dblquad(lambda _, x: x ** 2, 0, 1, lambda xu: Ydown(xu), lambda xd: Yup(xd),
                                               epsrel=epsrel)[0]
        self.__props['Jxy'] = integrate.dblquad(lambda y, x: x * y, 0, 1, lambda xu: Ydown(xu), lambda xd: Yup(xd),
                                                epsrel=epsrel)[0]
        self.__props['Jxc'] = self.__props['Jx'] - self.__props['a_b'] * self.__props['y0'] ** 2
        self.__props['Jyc'] = self.__props['Jy'] - self.__props['a_b'] * self.__props['x0'] ** 2
        self.__props['Jxcyc'] = self.__props['Jxy'] - self.__props['a_b'] * self.__props['x0'] * self.__props['y0']
        self.__props['Jp'] = self.__props['Jxc'] + self.__props['Jyc']
        self.__props['Wp'] = self.__props['Jp'] / max(
            sqrt((0 - self.__props['x0']) ** 2 + (0 - self.__props['y0']) ** 2),
            sqrt((1 - self.__props['x0']) ** 2 + (0 - self.__props['y0']) ** 2))
        self.__props['alpha'] = 0.5 * arctan(-2 * self.__props['Jxcyc'] / (self.__props['Jxc'] - self.__props['Jyc']))

        return self.__props

    def export(self, file_path='exports/airfoil', file_name='airfoil', file_extension='xlsx',
               show_time=True, header=True):
        export2(self.to_dataframe(bears='pandas'),
                file_path=file_path,
                file_name=file_name,
                file_extension=file_extension,
                sheet_name='airfoil',
                show_time=show_time,
                header=header)
        export2(pd.DataFrame(self.properties, index=[0]),
                file_path=file_path,
                file_name=file_name + '_properties',
                file_extension=file_extension,
                sheet_name=file_name + '_properties',
                show_time=show_time,
                header=header)


class Grate:

    def __init__(self, airfoil, gamma: float, t_b: int | float = 1.0, N: int = 20):
        self.__airfoil = airfoil
        self.__t_b = t_b  # относительный шаг []
        self.__gamma = gamma  # угол установки [рад]
        self.__N = N  #
        self.coords = dict()
        self.__props = dict()

    @property
    def t_b(self) -> float:
        return self.__t_b

    @t_b.setter
    def t_b(self, t_b):
        if type(t_b) in (int, float) and t_b > 0:
            self.__t_b = t_b
        else:
            print(Fore.RED + f't_b is float or int > 0!' + Fore.RESET)

    @t_b.deleter
    def t_b(self):
        self.reset()

    @property
    def gamma(self) -> float:
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma):
        if type(gamma) is float or type(gamma) is int:
            self.__gamma = gamma
        else:
            print(Fore.RED + f'gamma is float or int!' + Fore.RESET)

    @gamma.deleter
    def gamma(self):
        self.reset()

    def reset(self):
        self.__t_b = 1  # относительный шаг []
        self.__gamma = 0  # угол установки [рад]
        self.coords = dict()
        self.__props = dict()

    @timeit()
    def solve(self):
        self.coords['u'], self.coords['l'] = {'x': [], 'y': []}, {'x': [], 'y': []}

        dct = self.__airfoil.transform(self.__airfoil.properties['x0'], self.__airfoil.properties['y0'],
                                       self.gamma, scale=1, inplace=False)
        XuG, YuG, XdG, YdG = dct['u']['x'], dct['u']['y'], dct['l']['x'], dct['l']['y']
        del dct

        # отсечка значений спинки корыту и наоборот
        if self.__gamma >= 0:
            XuGt = list(reversed(XdG[1:XdG.index(min(XdG)) + 1])) + XuG[:XuG.index(max(XuG)) + 1]
            YuGt = list(reversed(YdG[1:XdG.index(min(XdG)) + 1])) + YuG[:XuG.index(max(XuG)) + 1]
            XdGt = XdG[XdG.index(min(XdG)):] + list(reversed(XuG[XuG.index(max(XuG)):len(XuG) - 1]))
            YdGt = YdG[XdG.index(min(XdG)):] + list(reversed(YuG[XuG.index(max(XuG)):len(XuG) - 1]))
        else:
            XuGt = XuG[XuG.index(min(XuG)):] + list(reversed(XdG[XdG.index(max(XdG)):len(XdG) - 1]))
            YuGt = YuG[XuG.index(min(XuG)):] + list(reversed(YdG[XdG.index(max(XdG)):len(XdG) - 1]))
            XdGt = list(reversed(XuG[1:XuG.index(min(XuG)) + 1])) + XdG[:XdG.index(max(XdG)) + 1]
            YdGt = list(reversed(YuG[1:XuG.index(min(XuG)) + 1])) + YdG[:XdG.index(max(XdG)) + 1]

        self.coords['u']['x'], self.coords['u']['y'] = XuGt, YuGt
        self.coords['l']['x'], self.coords['l']['y'] = XdGt, YdGt

        return self.coords

    @property
    @timeit()
    def properties(self, epsrel=0.01):
        """Дифузорность/конфузорность решетки"""
        if self.__props: return self.__props

        # kind='cubic' необходим для гладкости производной
        Fd = interpolate.interp1d(self.coords['l']['x'], [y + self.__t_b / 2 for y in self.coords['l']['y']],
                                  kind='cubic', fill_value='extrapolate')
        Fu = interpolate.interp1d(self.coords['u']['x'], [y - self.__t_b / 2 for y in self.coords['u']['y']],
                                  kind='cubic', fill_value='extrapolate')

        xgmin = min(self.coords['u']['x'] + self.coords['l']['x']) + self.__airfoil.r_inlet_b
        ygmin = min(self.coords['l']['y']) - self.__t_b / 2
        xgmax = max(self.coords['u']['x'] + self.coords['l']['x']) - self.__airfoil.r_outlet_b
        ygmax = max(self.coords['u']['y']) + self.__t_b / 2

        self.d, self.xd, self.yd = [], [], []
        xu, yu = [], []
        xd, yd = [], []
        Lu, Ld = [], []

        def dfdx(x0, F, dx=1e-6):
            return (F(x0 + dx / 2) - F(x0 - dx / 2)) / dx

        def ABC(x0, F):
            df_dx = dfdx(x0, F)
            return -1 / df_dx, -1, -(-1 / df_dx) * x0 - (-1) * F(x0)

        def cosfromtan(tg):
            return sqrt(1 / (tg ** 2 + 1))

        x = xgmin
        while True:
            if x >= xgmax: break
            xd.append(x)
            yd.append(Fd(x))
            Ld.append(ABC(x, Fd))
            x += epsrel * cosfromtan(dfdx(x, Fd))

        x = xgmin
        while True:
            if x >= xgmax: break
            xu.append(x)
            yu.append(Fu(x))
            Lu.append(ABC(x, Fu))
            x += epsrel ** 2 * cosfromtan(dfdx(x, Fd))

        j0 = 0  # начальный индекс искомого перпендикуляра
        for i in tqdm(range(len(Ld)), desc='Channel calculation'):
            epsilon = inf
            for j in range(j0, len(Lu)):

                if abs(eps('rel', Ld[i][0], Lu[j][0])) <= epsrel:
                    if dist2line(xd[i], yd[i], *Lu[j]) > epsrel: continue
                    self.d.append(dist((xd[i], yd[i]), (xu[j], yu[j])))
                    self.xd.append(0.5 * (xd[i] + xu[j]))
                    self.yd.append(0.5 * (yd[i] + yu[j]))
                    break

                xdt, ydt = COOR(Ld[i][0], Ld[i][2], Lu[j][0], Lu[j][2])  # точка пересечения перпендикуляров

                dd = dist((xdt, ydt), (xd[i], yd[i])) * 2
                du = dist((xdt, ydt), (xu[j], yu[j])) * 2

                abs_eps_rel = abs(eps('rel', dd, du))
                if abs_eps_rel < epsilon and ygmin < ydt < ygmax and xgmin <= xdt <= xgmax:
                    epsilon = abs_eps_rel
                    D = (dd + du) / 2
                    XDT, YDT = xdt, ydt
                    jt = j

            if epsilon < epsrel:
                self.d.append(D)
                self.xd.append(XDT)
                self.yd.append(YDT)
                j0 = jt

        '''self.r = [0]
        for i in range(len(self.d) - 1):
            self.r.append(self.r[i] + dist((self.xd[i], self.yd[i]), (self.xd[i + 1], self.yd[i + 1])))'''

        self.r = np.zeros(len(self.d))
        for i in range(1, len(self.d)):
            self.r[i] = self.r[i - 1] + dist((self.xd[i - 1], self.yd[i - 1]), (self.xd[i], self.yd[i]))

        self.__props = {tuple((self.xd[i], self.yd[i])): self.d[i] for i in range(len(self.d))}

        return self.__props

    def show(self):
        if not self.__props: self.properties
        fg = plt.figure(figsize=(12, 6))  # размер в дюймах
        gs = fg.add_gridspec(1, 2)  # строки, столбца

        fg.add_subplot(gs[0, 0])  # позиция графика
        plt.title('Lattice')
        plt.grid(True)
        plt.axis('square')
        plt.xlim(floor(min(self.coords['u']['x'] + self.coords['l']['x'])),
                 ceil(max(self.coords['u']['x'] + self.coords['l']['x'])))
        plt.ylim(-1, 1)
        for i in range(len(self.d)):
            plt.plot(list(self.d[i] / 2 * cos(alpha) + self.xd[i] for alpha in linspace(0, 2 * pi, 360)),
                     list(self.d[i] / 2 * sin(alpha) + self.yd[i] for alpha in linspace(0, 2 * pi, 360)),
                     ls='-', color=(0, 1, 0))
        plt.plot(self.xd, self.yd, ls='--', color=(1, 0, 1))
        plt.plot([min(self.coords['u']['x'] + self.coords['l']['x'])] * 2, [-1, 1],
                 [max(self.coords['u']['x'] + self.coords['l']['x'])] * 2, [-1, 1],
                 ls='-', color=(0, 0, 0))  # границы решетки
        plt.plot(self.coords['u']['x'], list(y - self.__t_b / 2 for y in self.coords['u']['y']),
                 ls='-', color=(0, 0, 0))
        plt.plot(self.coords['l']['x'], list(y - self.__t_b / 2 for y in self.coords['l']['y']),
                 ls='-', color=(0, 0, 0))
        plt.plot(self.coords['u']['x'], list(y + self.__t_b / 2 for y in self.coords['u']['y']),
                 ls='-', color=(0, 0, 0))
        plt.plot(self.coords['l']['x'], list(y + self.__t_b / 2 for y in self.coords['l']['y']),
                 ls='-', color=(0, 0, 0))

        fg.add_subplot(gs[0, 1])  # позиция графика
        plt.title('Channel')
        plt.grid(True)
        plt.axis('square')
        plt.xlim([0, ceil(max(self.r))])
        plt.ylim([0, ceil(self.__t_b)])
        plt.plot(self.r, self.d, ls='-', color=(0, 1, 0))
        plt.plot([0, ceil(max(self.r))], [0, 0], ls='-', color=(0, 0, 0), linewidth=1.5)
        plt.plot(list((self.r[i] + self.r[i + 1]) / 2 for i in range(len(self.r) - 1)),
                 list((self.d[i + 1] - self.d[i]) / (self.r[i + 1] - self.r[i]) for i in range(len(self.r) - 1)),
                 ls='-', color=(1, 0, 0))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    Airfoil.rnd = 4

    airfoils = list()

    if 1:
        airfoils.append(Airfoil('BMSTU', 20))

        airfoils[-1].xg_b = 0.35
        airfoils[-1].r_inlet_b = 0.06
        airfoils[-1].r_outlet_b = 0.03
        airfoils[-1].g_ = 0.5
        airfoils[-1].g_inlet = radians(25)
        airfoils[-1].g_outlet = radians(10)
        airfoils[-1].e = radians(110)

    if 1:
        airfoils.append(Airfoil('NACA', 40))

        airfoils[-1].c_b = 0.24
        airfoils[-1].f_b = 0.05
        airfoils[-1].xf_b = 0.3

    if 1:
        airfoils.append(Airfoil('MYNK', 20))

        airfoils[-1].h = 0.1

    if 1:
        airfoils.append(Airfoil('PARSEC', 30))

        airfoils[-1].r_inlet_b = 0.05
        airfoils[-1].r_inlet_b = 0.05

    if 0:
        airfoils.append(Airfoil('BEZIER', 30))

        airfoils[-1].u = ((0.0, 0.0), (0.05, 0.100), (0.35, 0.200), (1.0, 0.0))
        airfoils[-1].l = ((0.0, 0.0), (0.05, -0.10), (0.35, -0.05), (0.5, 0.0), (1.0, 0.0))

    for airfoil in airfoils:
        airfoil.solve()
        airfoil.show()

        print(airfoil.to_dataframe(bears='pandas'))
        print(airfoil.to_dataframe(bears='polars'))

        print(Fore.MAGENTA + 'airfoil properties:' + Fore.RESET)
        for k, v in airfoil.properties.items(): print(f'{k}: {v}')

        airfoil.export()

        grate = Grate(airfoil, radians(46.23), 1 / 1.698, N=20)  # относ. шаг профиля, угол установки профиля

        grate.solve()
        grate.show()

        # print(grate.to_dataframe())
        # print(grate.to_dataframe(bears='polars'))

        # print(Fore.MAGENTA + 'grate properties:' + Fore.RESET)
        # for k, v in grate.properties.items(): print(f'{k}: {v}')
