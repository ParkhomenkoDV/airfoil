"""
Список литературы:

[1] =
"""

import sys
import warnings
from tqdm import tqdm
from colorama import Fore

import numpy as np
from numpy import array, arange, linspace, zeros
from numpy import nan, isnan, inf, isinf, pi
from numpy import cos, sin, tan, arctan as atan, sqrt, floor, ceil, radians, degrees
from scipy import interpolate, integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from curves import bernstein_curve

sys.path.append('D:/Programming/Python/scripts')

from tools import export2, COOR, Axis, angle, rounding, eps, dist, dist2line, isiter, derivative, line_coefs, tan2cos
from decorators import timeit, warns


class Airfoil:
    """Относительный аэродинамический профиль"""

    rnd = 4  # количество значащих цифр
    __discreteness = 30  # рекомендуемое количество дискретных точек
    # TODO
    __methods = {'BMSTU': {'description': '',
                           'aliases': ('BMSTU', 'МГТУ', 'МВТУ', 'МИХАЛЬЦЕВ'),
                           'attributes': {}},
                 'NACA': {'description': '',
                          'aliases': ('NACA', 'N.A.C.A.'),
                          'attributes': {}},
                 'MYNK': {'description': '',
                          'aliases': ('MYNK', 'МУНК'),
                          'attributes': {}},
                 'PARSEC': {'description': '',
                            'aliases': ('PARSEC',),
                            'attributes': {}},
                 'BEZIER': {'description': '',
                            'aliases': ('BEZIER', 'БЕЗЬЕ'),
                            'attributes': {}},
                 'MANUAL': {'description': '',
                            'aliases': ('MANUAL', 'ВРУЧНУЮ'),
                            'attributes': {}}, }
    __relative_step = 1.0  # дефолтный относительный шаг []
    __gamma = 0.0  # дефолтный угол установки [рад]

    def validate(self, **kwargs) -> None:
        """Проверка верности ввода атрибутов профиля"""

        method = kwargs.pop('method', None)
        if method is not None:
            assert isinstance(method, str)
            method = method.strip().upper()
            assert any(method in value for value in Airfoil.__methods.values())

        discreteness = kwargs.pop('discreteness', None)
        if discreteness is not None:
            assert isinstance(discreteness, int) and 3 <= discreteness

        relative_step = kwargs.pop('relative_step', None)
        if relative_step is not None:
            assert isinstance(relative_step, (float, int, np.number)) and 0 < relative_step

        gamma = kwargs.pop('gamma', None)
        if gamma is not None:
            assert isinstance(gamma, (float, int, np.number)) and -pi / 2 <= gamma <= pi / 2

        def validate_points(points) -> None:
            """Проверка двумерного массива точек"""
            assert isiter(points)  # проверка на итератор
            assert all(map(isiter, points))  # проверка элементов итератора на итератор
            assert all(len(el) == 2 for el in points)  # проверка длин элементов итератора
            assert all(isinstance(el, (int, float)) for itr in points for el in itr)  # проверка типов элементов

        if hasattr(self, '_Airfoil__method'):
            if self.__method in Airfoil.__methods['BMSTU']:
                # относ. координата пересечения входного и выходного лучей
                assert hasattr(self, 'xg_b')
                assert isinstance(self.xg_b, (int, float))
                assert 0 <= self.xg_b <= 1

                # относ. радиус входной кромоки
                assert hasattr(self, 'r_inlet_b')
                assert isinstance(self.r_inlet_b, (int, float))
                assert 0 <= self.r_inlet_b <= 1

                # относ. радиус выходной кромки
                assert hasattr(self, 'r_outlet_b')
                assert isinstance(self.r_outlet_b, (int, float))
                assert 0 <= self.r_outlet_b <= 1

                # степень приближенности к спинке
                assert hasattr(self, 'g_')
                assert isinstance(self.g_, (int, float))
                assert 0 <= self.g_ <= 1

                # угол раскрытия входной кромоки
                assert hasattr(self, 'g_inlet')
                assert isinstance(self.g_inlet, (int, float, np.number))
                assert 0 <= self.g_inlet

                # угол раскрытия выходной кромки
                assert hasattr(self, 'g_outlet')
                assert isinstance(self.g_outlet, (int, float, np.number))
                assert 0 <= self.g_outlet

                # угол поворота потока
                assert hasattr(self, 'e')
                assert isinstance(self.e, (int, float, np.number))

            elif self.__method in Airfoil.__methods['NACA']:
                # относ. максимальная толщина профиля
                assert hasattr(self, 'c_b')
                assert isinstance(self.c_b, (int, float))
                assert 0 <= self.c_b <= 1

                # относ. координата максимального прогиба профиля
                assert hasattr(self, 'xf_b')
                assert isinstance(self.xf_b, (int, float))
                assert 0 <= self.xf_b <= 1

                # относ. максимальный прогиб профиля
                assert hasattr(self, 'f_b')
                assert isinstance(self.f_b, (int, float))
                assert 0 <= self.f_b <= 1

            elif self.__method in Airfoil.__methods['MYNK']:
                assert hasattr(self, 'h')
                assert isinstance(self.h, (int, float))
                assert 0 <= self.h <= 1

            elif self.__method in Airfoil.__methods['PARSEC']:
                # относ. радиус входной кромки
                assert hasattr(self, 'r_inlet_b')
                assert isinstance(self.r_inlet_b, (int, float))
                assert 0 <= self.r_inlet_b <= 1

                # относ. координата максимального прогиба спинки
                assert hasattr(self, "f_b_u")
                assert isinstance(self.f_b_u, (tuple, list))
                assert len(self.f_b_u) == 2
                assert all(isinstance(x, float) for x in self.f_b_u)

                # относ. координата максимального прогиба крыта
                assert hasattr(self, "f_b_l")
                assert isinstance(self.f_b_l, (tuple, list))
                assert len(self.f_b_l) == 2
                assert all(isinstance(x, float) for x in self.f_b_l)

                # кривизна спинки (вторая производная поверхности)
                assert hasattr(self, "d2y_dx2_u")
                assert isinstance(self.d2y_dx2_u, (float, int))

                # кривизна корыта (вторая производная поверхности)
                assert hasattr(self, "d2y_dx2_l")
                assert isinstance(self.d2y_dx2_l, (float, int))

                # угол выхода между поверхностью спинки и горизонталью [рад]
                assert hasattr(self, "theta_outlet_u")
                assert isinstance(self.theta_outlet_u, float)

                # угол выхода между поверхностью корыта и горизонталью [рад]
                assert hasattr(self, "theta_outlet_l")
                assert isinstance(self.theta_outlet_l, float)

            elif self.__method in Airfoil.__methods['BEZIER']:
                assert hasattr(self, 'u') and hasattr(self, 'l')
                validate_points(self.u)
                validate_points(self.l)

            elif self.__method in Airfoil.__methods['MANUAL']:
                assert hasattr(self, 'u') and hasattr(self, 'l')
                validate_points(self.u)
                validate_points(self.l)
                assert hasattr(self, 'deg') and isinstance(self.deg, int) and 0 <= self.deg <= 3

    def __init__(self, method: str, discreteness: int = __discreteness,
                 relative_step: float | int = __relative_step, gamma: float | int = __gamma):
        self.validate(method=method, discreteness=discreteness, relative_step=relative_step, gamma=gamma)

        self.__method = method  # метод построения аэродинамического профиля
        self.__discreteness = discreteness  # количество точек дискретизации

        self.__relative_step = relative_step  # относительный шаг []
        self.__gamma = gamma  # угол установки [рад]

        self.__coords = {'u': {'x': list(), 'y': list()},  # относительные координаты спинки
                         'l': {'x': list(), 'y': list()}}  # относительные координаты корыта
        self.__props = dict()  # относительные характеристики профиля
        self.__channel = dict()  # дифузорность/конфузорность решетки

    def __str__(self) -> str:
        return self.__method

    @property
    def method(self) -> str:
        return self.__method

    @method.setter
    def method(self, method: str) -> None:
        self.validate(method=method)
        self.__init__(method)  # снос предыдущих расчетов для нового метода

    @method.deleter
    def method(self) -> None:
        raise

    @property
    def discreteness(self) -> int:
        return self.__discreteness

    @discreteness.setter
    def discreteness(self, discreteness) -> None:
        self.validate(discreteness=discreteness)
        self.__init__(method=self.method, discreteness=discreteness)

    @discreteness.deleter
    def discreteness(self) -> None:
        raise

    @property
    def relative_step(self) -> float | int | np.number:
        return self.__relative_step

    @relative_step.setter
    def relative_step(self, relative_step):
        self.validate(relative_step=relative_step)
        self.__relative_step = relative_step
        self.__channel = dict()  # дифузорность/конфузорность решетки

    @relative_step.deleter
    def relative_step(self):
        self.__relative_step = Airfoil.__relative_step

    @property
    def gamma(self) -> float | int | np.number:
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma):
        self.validate(gamma=gamma)
        self.__init__(method=self.method, gamma=gamma)

    @gamma.deleter
    def gamma(self):
        self.__gamma = Airfoil.__gamma

    @property
    def coords(self) -> dict[str:dict]:
        return self.__coords

    # TODO
    def input(self):
        """Динамический ввод с защитой от дураков"""
        pass

    def __bmstu(self) -> None:
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
        xcl_u, ycl_u = COOR(tan(atan(k_inlet) + g_u_inlet),
                            sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * self.r_inlet_b -
                            (tan(atan(k_inlet) + g_u_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1],
                            tan(atan(k_outlet) - g_u_outlet),
                            sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * self.r_outlet_b -
                            (tan(atan(k_outlet) - g_u_outlet)) * self.__O_outlet[0] - (-1) * self.__O_outlet[1])

        xcl_d, ycl_d = COOR(tan(atan(k_inlet) - g_d_inlet),
                            -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * self.r_inlet_b -
                            (tan(atan(k_inlet) - g_d_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1],
                            tan(atan(k_outlet) + g_d_outlet),
                            -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * self.r_outlet_b -
                            (tan(atan(k_outlet) + g_d_outlet)) * self.__O_outlet[0] - (-1) * self.__O_outlet[1])

        # точки пересечения окружностей со спинкой и корытом
        xclc_i_u, yclc_i_u = COOR(tan(atan(k_inlet) + g_u_inlet),
                                  sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * self.r_inlet_b
                                  - (tan(atan(k_inlet) + g_u_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1],
                                  -1 / (tan(atan(k_inlet) + g_u_inlet)),
                                  -(-1 / tan(atan(k_inlet) + g_u_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1])

        xclc_i_d, yclc_i_d = COOR(tan(atan(k_inlet) - g_d_inlet),
                                  -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * self.r_inlet_b
                                  - (tan(atan(k_inlet) - g_d_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1],
                                  -1 / (tan(atan(k_inlet) - g_d_inlet)),
                                  -(-1 / tan(atan(k_inlet) - g_d_inlet)) * self.__O_inlet[0] - (-1) * self.__O_inlet[1])

        xclc_e_u, yclc_e_u = COOR(tan(atan(k_outlet) - g_u_outlet),
                                  sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * self.r_outlet_b
                                  - tan(atan(k_outlet) - g_u_outlet) * self.__O_outlet[0] - (-1) * self.__O_outlet[1],
                                  -1 / tan(atan(k_outlet) - g_u_outlet),
                                  -(-1 / tan(atan(k_outlet) - g_u_outlet)) * self.__O_outlet[0] - (-1) *
                                  self.__O_outlet[1])

        xclc_e_d, yclc_e_d = COOR(tan(atan(k_outlet) + g_d_outlet),
                                  -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * self.r_outlet_b
                                  - tan(atan(k_outlet) + g_d_outlet) * self.__O_outlet[0] - (-1) * self.__O_outlet[1],
                                  -1 / tan(atan(k_outlet) + g_d_outlet),
                                  -(-1 / tan(atan(k_outlet) + g_d_outlet)) * self.__O_outlet[0] - (-1) *
                                  self.__O_outlet[1])

        # точки входной окружности кромки по спинке
        an = angle(points=((0, self.__O_inlet[1]), self.__O_inlet, (xclc_i_u, yclc_i_u)))
        if xclc_i_u > self.__O_inlet[0]: an = pi - an
        # уменьшение угла для предотвращения дублирования координат
        angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
        self.coords['u']['x'] = (self.r_inlet_b * (1 - cos(angles))).tolist()
        self.coords['u']['y'] = (self.__O_inlet[1] + self.r_inlet_b * sin(angles)).tolist()

        # точки входной окружности кромки по корыту
        an = angle(points=((0, self.__O_inlet[1]), self.__O_inlet, (xclc_i_d, yclc_i_d)))
        if xclc_i_d > self.__O_inlet[0]: an = pi - an
        # уменьшение угла для предотвращения дублирования координат
        angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
        self.coords['l']['x'] = (self.r_inlet_b * (1 - cos(angles))).tolist()
        self.coords['l']['y'] = (self.__O_inlet[1] - self.r_inlet_b * sin(angles)).tolist()

        xu, yu = bernstein_curve(((xclc_i_u, yclc_i_u), (xcl_u, ycl_u), (xclc_e_u, yclc_e_u)),
                                 N=self.__discreteness).T.tolist()
        xd, yd = bernstein_curve(((xclc_i_d, yclc_i_d), (xcl_d, ycl_d), (xclc_e_d, yclc_e_d)),
                                 N=self.__discreteness).T.tolist()
        self.coords['u']['x'] += xu
        self.coords['u']['y'] += yu
        self.coords['l']['x'] += xd
        self.coords['l']['y'] += yd

        # точки выходной окружности кромки по спинке
        an = angle(points=((1, self.__O_outlet[1]), self.__O_outlet, (xclc_e_u, yclc_e_u)))
        if self.__O_outlet[0] > xclc_e_u: an = pi - an
        # уменьшение угла для предотвращения дублирования координат
        angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
        self.coords['u']['x'] += (1 - self.r_outlet_b * (1 - cos(angles))).tolist()[::-1]
        self.coords['u']['y'] += (self.__O_outlet[1] + self.r_outlet_b * sin(angles)).tolist()[::-1]

        # точки выходной окружности кромки по корыту
        an = angle(points=((1, self.__O_outlet[1]), self.__O_outlet, (xclc_e_d, yclc_e_d)))
        if self.__O_outlet[0] > xclc_e_d: an = pi - an
        # уменьшение угла для предотвращения дублирования координат
        angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
        self.coords['l']['x'] += (1 - self.r_outlet_b * (1 - cos(angles))).tolist()[::-1]
        self.coords['l']['y'] += (self.__O_outlet[1] - self.r_outlet_b * sin(angles)).tolist()[::-1]

    def __naca(self, closed=True) -> None:
        i = arange(self.__N)
        betta = i * pi / (2 * (self.__N - 1))
        x = 1 - cos(betta)

        mask = (0 <= x) & (x <= self.xf_b)

        yf = np.full_like(i, self.f_b, dtype=np.float64)
        yf[mask] *= self.xf_b ** (-2) * (2 * self.xf_b * x[mask] - x[mask] ** 2)
        yf[~mask] *= (1 - self.xf_b) ** (-2) * (1 - 2 * self.xf_b + 2 * self.xf_b * x[~mask] - x[~mask] ** 2)

        gradYf = 2 * self.f_b

        a = array((0.2969, -0.126, -0.3516, 0.2843, -0.1036 if closed else -0.1015), dtype='float64')

        yc = self.c_b / 0.2 * np.dot(a, np.column_stack((sqrt(x), x, x ** 2, x ** 3, x ** 4)).T)

        tetta = atan(gradYf)

        sin_tetta, cos_tetta = sin(tetta), cos(tetta)  # предварительный расчет для ускорения работы

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

    def __mynk(self) -> None:
        x = linspace(0, 1, self.__N)
        self.coords['u']['x'] = x
        self.coords['u']['y'] = self.h * (0.25 * (-x - 17 * x ** 2 - 6 * x ** 3) + x ** 0.87 * (1 - x) ** 0.56)
        self.coords['l']['x'] = x
        self.coords['l']['y'] = self.h * (0.25 * (-x - 17 * x ** 2 - 6 * x ** 3) - x ** 0.87 * (1 - x) ** 0.56)

        angle = atan((self.coords['u']['y'][-1] - self.coords['u']['y'][0]) / (1 - 0))
        scale = dist((self.coords['u']['x'][0], self.coords['u']['y'][0]),
                     (self.coords['u']['x'][-1], self.coords['u']['y'][-1]))
        self.transform(angle=angle, scale=1 / scale, inplace=True)

        self.find_circles()

    def __parsec_coefficients(self, surface: str,
                              radius_inlet: float | int,
                              c_b: tuple[float, float], d2y_dx2_surface,
                              outlet: tuple, theta_outlet_surface: float | int):
        """PARSEC coefficients"""
        assert surface in ('l', 'u')
        assert isinstance(c_b, (tuple, list)) and len(c_b) == 2
        assert isinstance(outlet, (tuple, list)) and len(outlet) == 2

        x_c_b, y_c_b = c_b
        x_outlet, y_outlet = outlet

        coef = zeros(6)

        # 1-ый коэффициент зависит от кривой поверхности спинки или корыта
        coef[0] = -sqrt(2 * radius_inlet) if surface == 'l' else sqrt(2 * radius_inlet)

        i = arange(1, 6)

        # матрицы коэффициентов системы уравнений
        A = array([x_outlet ** (i + 0.5),
                   x_c_b ** (i + 0.5),
                   (i + 0.5) * x_outlet ** (i - 0.5),
                   (i + 0.5) * x_c_b ** (i - 0.5),
                   (i ** 2 - 0.25) * x_c_b ** (i - 1.5)])
        B = array([[y_outlet - coef[0] * sqrt(x_outlet)],
                   [y_c_b - coef[0] * sqrt(x_c_b)],
                   [tan(theta_outlet_surface) - 0.5 * coef[0] * (1 / sqrt(x_outlet))],
                   [-0.5 * coef[0] * (1 / sqrt(x_c_b))],
                   [d2y_dx2_surface + 0.25 * coef[0] * x_c_b ** (-1.5)]])

        X = np.linalg.solve(A, B)  # решение СЛАУ
        coef[1:6] = X[0:5, 0]

        return coef

    def __parsec(self) -> None:
        """
        Generate and plot the contour of an airfoil using the PARSEC parameterization
        H. Sobieczky, *'Parametric airfoils and wings'* in *Notes on Numerical Fluid Mechanics*, Vol. 68, pp 71-88]
        (www.as.dlr.de/hs/h-pdf/H141.pdf)
        Repository & documentation: http://github.com/dqsis/parsec-airfoils
        """

        # поверхностные коэффициенты давления спинки и корыта
        cf_u = self.__parsec_coefficients('u', self.r_inlet_b, self.f_b_u, self.d2y_dx2_u, (1, 0), self.theta_outlet_u)
        cf_l = self.__parsec_coefficients('l', self.r_inlet_b, self.f_b_l, self.d2y_dx2_l, (1, 0), self.theta_outlet_l)

        self.coords['u']['x'] = linspace(0, 1, self.__N)
        self.coords['u']['y'] = sum([cf_u[i] * self.coords['u']['x'] ** (i + 0.5) for i in range(6)])
        self.coords['l']['x'] = linspace(1, 0, self.__N)
        self.coords['l']['y'] = sum([cf_l[i] * self.coords['l']['x'] ** (i + 0.5) for i in range(6)])
        self.coords['l']['x'], self.coords['l']['y'] = self.coords['l']['x'][::-1], self.coords['l']['y'][::-1]

        self.__O_inlet, self.r_inlet_b = (0, 0), 0
        self.__O_outlet, self.r_outlet_b = (1, 0), 0

    def __bezier(self) -> None:
        if not any(p[0] == 0 for p in self.u): self.u = list(self.u) + [(0, 0)]
        if not any(p[0] == 1 for p in self.u): self.u = list(self.u) + [(1, 0)]

        self.coords['u']['x'], self.coords['u']['y'] = bernstein_curve(self.u, N=self.__N).T
        self.coords['l']['x'], self.coords['l']['y'] = bernstein_curve(self.l, N=self.__N).T

        self.__O_inlet, self.r_inlet_b = (0, 0), 0
        self.__O_outlet, self.r_outlet_b = (1, 0), 0

    def __manual(self) -> None:
        if not any(p[0] == 0 for p in self.u): self.u = list(self.u) + [(0, 0)]
        if not any(p[0] == 1 for p in self.u): self.u = list(self.u) + [(1, 0)]

        x = linspace(0, 1, self.__N)
        self.coords['u']['x'], self.coords['u']['y'] = x, interpolate.interp1d([p[0] for p in self.u],
                                                                               [p[1] for p in self.u],
                                                                               kind=self.deg)(x)
        self.coords['l']['x'], self.coords['l']['y'] = x, interpolate.interp1d([p[0] for p in self.l],
                                                                               [p[1] for p in self.l],
                                                                               kind=self.deg)(x)

        self.__O_inlet, self.r_inlet_b = (0, 0), 0
        self.__O_outlet, self.r_outlet_b = (1, 0), 0

    @timeit()
    def solve(self):
        self.__props = dict()
        self.coords['u'], self.coords['l'] = {'x': list(), 'y': list()}, {'x': list(), 'y': list()}
        self.validate()

        if self.method.upper() in ('NACA', 'N.A.C.A.'):
            self.__naca()
        elif self.method.upper() in ('BMSTU', 'МГТУ', 'МВТУ', 'МИХАЛЬЦЕВ'):
            self.__bmstu()
        elif self.method.upper() in ('MYNK', 'МУНК'):
            self.__mynk()
        elif self.method.upper() in ('PARSEC',):
            self.__parsec()
        elif self.method.upper() in ('BEZIER', 'БЕЗЬЕ'):
            self.__bezier()
        elif self.method.upper() in ('MANUAL', 'ВРУЧНУЮ'):
            self.__manual()
        else:
            print(Fore.RED + f'No such method {self.method}!' + Fore.RESET)

        dct = self.transform(self.properties['x0'], self.properties['y0'],
                             self.__gamma, scale=1, inplace=False)
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

        x_min = min(np.min(self.coords['u']['x']), np.min(self.coords['l']['x']))
        x_max = max(np.max(self.coords['u']['x']), np.max(self.coords['l']['x']))
        scale = abs(x_max - x_min)
        self.transform(x0=x_min, scale=1 / scale, inplace=True)

        return self.coords

    def transform(self, x0=0.0, y0=0.0, angle=0.0, scale=1.0, inplace: bool = False) -> dict[str: dict]:
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

        Fu = interpolate.interp1d(*self.coords['u'].values(), kind='cubic', fill_value='extrapolate')
        Fd = interpolate.interp1d(*self.coords['l'].values(), kind='cubic', fill_value='extrapolate')

        dx = 0.000_1

        x0, x1 = 0, 1
        y0, y1 = Fu(x0), Fu(x1)

        x0u, x1u = dx, 1 - dx
        y0u, y1u = Fu(x0u), Fu(x1u)

        x0d, x1d = dx, 1 - dx
        y0d, y1d = Fd(x0d), Fd(x1d)

        A0u, B0u, C0u = line_coefs(p1=(x0, y0), p2=(x0u, y0u))
        A0d, B0d, C0d = line_coefs(p1=(x0, y0), p2=(x0d, y0d))
        A1u, B1u, C1u = line_coefs(p1=(x1, y1), p2=(x1u, y1u))
        A1d, B1d, C1d = line_coefs(p1=(x1, y1), p2=(x1d, y1d))

        # A для перпендикуляров
        AA0u, AA0d = -1 / A0u, -1 / A0d
        AA1u, AA1d = -1 / A1u, -1 / A1d

        # С для перпендикуляров
        CC0u, CC0d = y0u - AA0u * x0u, y0d - AA0d * x0d
        CC1u, CC1d = y1u - AA1u * x1u, y1d - AA1d * x1d

        # центры входной и выходной окружностей
        if not hasattr(self, '_Airfoil__O_inlet'): self.__O_inlet = COOR(AA0u, CC0u, AA0d, CC0d)
        if not hasattr(self, '_Airfoil__O_outlet'): self.__O_outlet = COOR(AA1u, CC1u, AA1d, CC1d)
        if not hasattr(self, 'r_inlet_b'): self.r_inlet_b = abs(self.__O_inlet[0] - x0)
        if not hasattr(self, 'r_outlet_b'): self.r_outlet_b = abs(self.__O_outlet[0] - x1)

        return {'inlet': {'O': self.__O_inlet, 'r': self.r_inlet_b},
                'outlet': {'O': self.__O_outlet, 'r': self.r_outlet_b}}

    def show(self, figsize=(12, 5.25), savefig=False):
        """Построение профиля"""
        fg = plt.figure(figsize=figsize)
        gs = fg.add_gridspec(1, 4)  # строки, столбцы

        fg.add_subplot(gs[0, 0])
        plt.title('Initial data')
        plt.grid(False)
        plt.axis('off')

        plt.plot([], label=f'method = {self.method}')
        plt.plot([], label=f'discreteness = {self.__discreteness}')
        for key, value in self.__dict__.items():
            if '__' not in key and type(value) in (int, float):
                plt.plot([], label=f'{key} = {rounding(value, self.rnd)}')
        plt.legend(loc='upper center')

        fg.add_subplot(gs[0, 1])
        plt.title('Airfoil structure')
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

    def to_array(self, duplicates: bool = True):
        """Перевод координат в массив обхода против часовой стрелки считая с выходной кромки"""
        assert isinstance(duplicates, bool)
        if duplicates:
            return array((self.coords['u']['x'][::-1] + self.coords['l']['x'],
                          self.coords['u']['y'][::-1] + self.coords['l']['y']),
                         dtype='float64').T
        else:
            return array((self.coords['u']['x'][::-1] + self.coords['l']['x'][1::],
                          self.coords['u']['y'][::-1] + self.coords['l']['y'][1::]),
                         dtype='float64').T

    def to_dataframe(self, bears: str = 'pandas'):
        if bears.strip().lower() == 'pandas':
            return pd.DataFrame({'xu': pd.Series(self.coords['u']['x']), 'yu': pd.Series(self.coords['u']['y']),
                                 'xd': pd.Series(self.coords['l']['x']), 'yd': pd.Series(self.coords['l']['y'])})
        if bears.strip().lower() == 'polars':
            return pl.concat([pl.DataFrame({'xu': self.coords['u']['x'], 'yu': self.coords['u']['y']}),
                              pl.DataFrame({'xl': self.coords['l']['x'], 'yl': self.coords['l']['y']})],
                             how='horizontal')
        print(Fore.RED + 'Unknown bears!' + Fore.RESET)
        print('Use "pandas" or "polars"!')

    @property
    @timeit()
    def properties(self, epsrel: float = 1e-4) -> dict[str: float]:
        if self.__props: return self.__props

        Yu = interpolate.interp1d(*self.coords['u'].values(), kind='cubic', fill_value='extrapolate')
        Yl = interpolate.interp1d(*self.coords['l'].values(), kind='cubic', fill_value='extrapolate')

        self.__props['a_b'] = integrate.dblquad(lambda _, __: 1, 0, 1, lambda xu: Yl(xu), lambda xd: Yu(xd),
                                                epsrel=epsrel)[0]
        self.__props['xc_b'], self.__props['c_b'] = -1.0, 0
        self.__props['xf_b'], self.__props['f_b'] = -1.0, 0
        for x in linspace(0, 1, int(ceil(1 / epsrel))):
            if Yu(x) - Yl(x) > self.__props['c_b']:
                self.__props['xc_b'], self.__props['c_b'] = x, Yu(x) - Yl(x)
            if abs((Yu(x) + Yl(x)) / 2) > abs(self.__props['f_b']):
                self.__props['xf_b'], self.__props['f_b'] = x, (Yu(x) + Yl(x)) / 2
        self.__props['Sx'] = integrate.dblquad(lambda y, _: y, 0, 1, lambda xu: Yl(xu), lambda xd: Yu(xd),
                                               epsrel=epsrel)[0]
        self.__props['Sy'] = integrate.dblquad(lambda _, x: x, 0, 1, lambda xu: Yl(xu), lambda xd: Yu(xd),
                                               epsrel=epsrel)[0]
        self.__props['x0'] = self.__props['Sy'] / self.__props['a_b']
        self.__props['y0'] = self.__props['Sx'] / self.__props['a_b']
        self.__props['Jx'] = integrate.dblquad(lambda y, _: y ** 2, 0, 1, lambda xu: Yl(xu), lambda xd: Yu(xd),
                                               epsrel=epsrel)[0]
        self.__props['Jy'] = integrate.dblquad(lambda _, x: x ** 2, 0, 1, lambda xu: Yl(xu), lambda xd: Yu(xd),
                                               epsrel=epsrel)[0]
        self.__props['Jxy'] = integrate.dblquad(lambda y, x: x * y, 0, 1, lambda xu: Yl(xu), lambda xd: Yu(xd),
                                                epsrel=epsrel)[0]
        self.__props['Jxc'] = self.__props['Jx'] - self.__props['a_b'] * self.__props['y0'] ** 2
        self.__props['Jyc'] = self.__props['Jy'] - self.__props['a_b'] * self.__props['x0'] ** 2
        self.__props['Jxcyc'] = self.__props['Jxy'] - self.__props['a_b'] * self.__props['x0'] * self.__props['y0']
        self.__props['Jp'] = self.__props['Jxc'] + self.__props['Jyc']
        self.__props['Wp'] = self.__props['Jp'] / max(
            sqrt((0 - self.__props['x0']) ** 2 + (0 - self.__props['y0']) ** 2),
            sqrt((1 - self.__props['x0']) ** 2 + (0 - self.__props['y0']) ** 2))
        self.__props['alpha'] = 0.5 * atan(-2 * self.__props['Jxcyc'] / (self.__props['Jxc'] - self.__props['Jyc']))
        self.__props['len_u'] = integrate.quad(lambda x: sqrt(1 + derivative(Yu, x) ** 2), 0, 1, epsrel=epsrel)[0]
        self.__props['len_l'] = integrate.quad(lambda x: sqrt(1 + derivative(Yl, x) ** 2), 0, 1, epsrel=epsrel)[0]

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

    '''
    @property
    @timeit()
    def grate(self, epsrel=0.01):
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

        self.d, self.xd, self.yd = list(), list(), list()
        xu, yu = list(), list()
        xd, yd = list(), list()
        Lu, Ld = list(), list()

        def dfdx(x0, F, dx: float = 1e-6):
            return (F(x0 + dx / 2) - F(x0 - dx / 2)) / dx

        def ABC(x0, F):
            """Коэффициенты A, B, C прямой"""
            df_dx = dfdx(x0, F)
            return -1 / df_dx, -1, -(-1 / df_dx) * x0 - (-1) * F(x0)

        cosfromtan = lambda tg: sqrt(1 / (tg ** 2 + 1))

        x = xgmin
        while x < xgmax:
            xd.append(x)
            yd.append(Fd(x))
            Ld.append(ABC(x, Fd))
            x += epsrel * cosfromtan(dfdx(x, Fd))

        x = xgmin
        while x < xgmax:
            xu.append(x)
            yu.append(Fu(x))
            Lu.append(ABC(x, Fu))
            x += (epsrel ** 2) * cosfromtan(dfdx(x, Fd))  # более мелкий шаг для лучшей дискретизации

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

                du, dd = dist((xdt, ydt), (xu[j], yu[j])) * 2, dist((xdt, ydt), (xd[i], yd[i])) * 2  # диаметры

                abs_eps_rel = abs(eps('rel', dd, du))
                if abs_eps_rel < epsilon and ygmin < ydt < ygmax and xgmin <= xdt <= xgmax:
                    epsilon = abs_eps_rel
                    D = 0.5 * (dd + du)
                    XDT, YDT = xdt, ydt
                    jt = j

            if epsilon < epsrel:
                self.d.append(D)
                self.xd.append(XDT)
                self.yd.append(YDT)
                j0 = jt

        self.r = zeros(len(self.d))
        for i in range(1, len(self.d)):
            self.r[i] = self.r[i - 1] + dist((self.xd[i - 1], self.yd[i - 1]), (self.xd[i], self.yd[i]))

        self.__props = {tuple((self.xd[i], self.yd[i])): self.d[i] for i in range(len(self.d))}

        return self.__props
    '''

    @property
    @timeit()
    def grate(self):
        """Дифузорность/конфузорность решетки"""
        if self.__channel: return self.__channel

        # kind='cubic' необходим для гладкости производной
        Fd = interpolate.interp1d(self.coords['l']['x'], [y + self.__t_b / 2 for y in self.coords['l']['y']], kind=3,
                                  fill_value='extrapolate')
        Fu = interpolate.interp1d(self.coords['u']['x'], [y - self.__t_b / 2 for y in self.coords['u']['y']], kind=3,
                                  fill_value='extrapolate')

        xgmin = min(self.coords['u']['x'] + self.coords['l']['x']) + self.r_inlet_b
        ygmin = min(self.coords['l']['y']) - self.__t_b / 2
        xgmax = max(self.coords['u']['x'] + self.coords['l']['x']) - self.r_outlet_b
        ygmax = max(self.coords['u']['y']) + self.__t_b / 2

        # длина кривой
        l = integrate.quad(lambda x: sqrt(1 + derivative(Fd, x) ** 2), xgmin, xgmax, limit=self.__N ** 2)[0]
        step = l / self.__discreteness

        x = [xgmin]
        while True:
            X = x[-1] + step * tan2cos(derivative(Fd, x[-1]))
            if X > xgmax: break
            x.append(X)
        x = array(x)

        Au, _, Cu = line_coefs(func=Fd, x0=x)

        def equations(vars, *args):
            """СНЛАУ"""
            x0, y0, r0, xl = vars
            xu, yu, Au, Cu = args

            Al, _, Cl = line_coefs(func=Fu, x0=xl)

            return [abs(Au * x0 + (-1) * y0 + Cu) / sqrt(Au ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xu - x0) ** 2 + (yu - y0) ** 2) - r0 ** 2,  # до кривой корыта
                    abs(Al * x0 + (-1) * y0 + Cl) / sqrt(Al ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xl - x0) ** 2 + (Fu(xl) - y0) ** 2) - r0 ** 2]  # до кривой спинки

        self.d, self.xd, self.yd = list(), list(), list()

        warnings.filterwarnings('error')
        for xu, yu, a_u, c_u in tqdm(zip(x, Fd(x), Au, Cu), desc='Channel calculation', total=len(x)):
            try:
                res = fsolve(equations, array((xu, yu, self.__t_b / 2, xu)), args=(xu, yu, a_u, c_u))
            except Exception:
                continue

            if xgmin <= res[0] <= xgmax and xgmin <= res[3] <= xgmax and res[2] <= self.__t_b / 2:
                self.d.append(res[2] * 2)
                self.xd.append(res[0])
                self.yd.append(res[1])
        warnings.filterwarnings('default')

        self.r = np.zeros(len(self.d))
        for i in range(1, len(self.d)):
            self.r[i] = self.r[i - 1] + dist((self.xd[i - 1], self.yd[i - 1]), (self.xd[i], self.yd[i]))

        self.__channel = {tuple((self.xd[i], self.yd[i])): self.d[i] for i in range(len(self.d))}

        return self.__channel

    def show_grate(self, n: int = 2):
        """Построение решетки"""
        assert isinstance(n, int) and 2 <= n  # количество профилей

        if not self.__channel: self.grate
        fg = plt.figure(figsize=(12, 6))  # размер в дюймах
        gs = fg.add_gridspec(1, 2)  # строки, столбца

        fg.add_subplot(gs[0, 0])  # позиция графика
        plt.title('Lattice')
        plt.grid(True)
        plt.axis('square')
        plt.xlim(floor(min(self.coords['u']['x'] + self.coords['l']['x'])),
                 ceil(max(self.coords['u']['x'] + self.coords['l']['x'])))
        plt.ylim(-1, 1)
        plt.plot([min(self.coords['u']['x'] + self.coords['l']['x'])] * 2, [-1, 1],
                 [max(self.coords['u']['x'] + self.coords['l']['x'])] * 2, [-1, 1],
                 ls='-', color='black')  # границы решетки
        for i in range(len(self.d)):
            plt.plot(list(self.d[i] / 2 * cos(alpha) + self.xd[i] for alpha in linspace(0, 2 * pi, 360)),
                     list(self.d[i] / 2 * sin(alpha) + self.yd[i] for alpha in linspace(0, 2 * pi, 360)),
                     ls='solid', color='green')
        plt.plot(self.xd, self.yd, ls='dashdot', color='orange',
                 label=f'gamma = {self.__gamma:.4f} [rad] = {degrees(self.__gamma):.4f} [deg]')
        plt.plot(self.coords['u']['x'], list(y - self.__t_b / 2 for y in self.coords['u']['y']),
                 ls='-', color='black', label=f't/b = {self.__t_b:.4f}')
        plt.plot(self.coords['l']['x'], list(y - self.__t_b / 2 for y in self.coords['l']['y']),
                 ls='-', color='black')
        plt.plot(self.coords['u']['x'], list(y + self.__t_b / 2 for y in self.coords['u']['y']),
                 ls='-', color='black')
        plt.plot(self.coords['l']['x'], list(y + self.__t_b / 2 for y in self.coords['l']['y']),
                 ls='-', color='black')
        plt.legend(fontsize=12)

        fg.add_subplot(gs[0, 1])  # позиция графика
        plt.title('Channel')
        plt.grid(True)
        plt.axis('square')
        plt.xlim([0, ceil(max(self.r))])
        plt.ylim([0, ceil(self.__t_b)])
        plt.plot(self.r, self.d, ls='-', color='green', label='channel')
        plt.plot([0, ceil(max(self.r))], [0, 0], ls='-', color=(0, 0, 0), linewidth=1.5)
        plt.plot(list((self.r[i] + self.r[i + 1]) / 2 for i in range(len(self.r) - 1)),
                 list((self.d[i + 1] - self.d[i]) / (self.r[i + 1] - self.r[i]) for i in range(len(self.r) - 1)),
                 ls='-', color='red', label='d2f/dx2')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


class Grate:
    """Аэродинамическая решетка"""

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
        self.coords['u'], self.coords['l'] = {'x': list(), 'y': list()}, {'x': list(), 'y': list()}

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

        self.d, self.xd, self.yd = list(), list(), list()
        xu, yu = list(), list()
        xd, yd = list(), list()
        Lu, Ld = list(), list()

        def dfdx(x0, F, dx: float = 1e-6):
            return (F(x0 + dx / 2) - F(x0 - dx / 2)) / dx

        def ABC(x0, F):
            """Коэффициенты A, B, C прямой"""
            df_dx = dfdx(x0, F)
            return -1 / df_dx, -1, -(-1 / df_dx) * x0 - (-1) * F(x0)

        cosfromtan = lambda tg: sqrt(1 / (tg ** 2 + 1))

        x = xgmin
        while x < xgmax:
            xd.append(x)
            yd.append(Fd(x))
            Ld.append(ABC(x, Fd))
            x += epsrel * cosfromtan(dfdx(x, Fd))

        x = xgmin
        while x < xgmax:
            xu.append(x)
            yu.append(Fu(x))
            Lu.append(ABC(x, Fu))
            x += (epsrel ** 2) * cosfromtan(dfdx(x, Fd))  # более мелкий шаг для лучшей дискретизации

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

                du, dd = dist((xdt, ydt), (xu[j], yu[j])) * 2, dist((xdt, ydt), (xd[i], yd[i])) * 2  # диаметры

                abs_eps_rel = abs(eps('rel', dd, du))
                if abs_eps_rel < epsilon and ygmin < ydt < ygmax and xgmin <= xdt <= xgmax:
                    epsilon = abs_eps_rel
                    D = 0.5 * (dd + du)
                    XDT, YDT = xdt, ydt
                    jt = j

            if epsilon < epsrel:
                self.d.append(D)
                self.xd.append(XDT)
                self.yd.append(YDT)
                j0 = jt

        self.r = zeros(len(self.d))
        for i in range(1, len(self.d)):
            self.r[i] = self.r[i - 1] + dist((self.xd[i - 1], self.yd[i - 1]), (self.xd[i], self.yd[i]))

        self.__props = {tuple((self.xd[i], self.yd[i])): self.d[i] for i in range(len(self.d))}

        return self.__props

    @property
    @timeit()
    def properties(self):
        """Дифузорность/конфузорность решетки"""
        if self.__props: return self.__props

        # kind='cubic' необходим для гладкости производной
        Fd = interpolate.interp1d(self.coords['l']['x'], [y + self.__t_b / 2 for y in self.coords['l']['y']], kind=3,
                                  fill_value='extrapolate')
        Fu = interpolate.interp1d(self.coords['u']['x'], [y - self.__t_b / 2 for y in self.coords['u']['y']], kind=3,
                                  fill_value='extrapolate')

        xgmin = min(self.coords['u']['x'] + self.coords['l']['x']) + self.__airfoil.r_inlet_b
        ygmin = min(self.coords['l']['y']) - self.__t_b / 2
        xgmax = max(self.coords['u']['x'] + self.coords['l']['x']) - self.__airfoil.r_outlet_b
        ygmax = max(self.coords['u']['y']) + self.__t_b / 2

        cosfromtan = lambda tg: sqrt(1 / (tg ** 2 + 1))

        # длина кривой
        l = integrate.quad(lambda x: sqrt(1 + derivative(Fd, x) ** 2), xgmin, xgmax, limit=self.__N ** 2)[0]
        step = l / self.__N

        x = [xgmin]
        while True:
            X = x[-1] + step * cosfromtan(derivative(Fd, x[-1]))
            if X > xgmax: break
            x.append(X)
        x = array(x)

        Au, _, Cu = line_coefs(func=Fd, x0=x)

        def equations(vars, *args):
            """СНЛАУ"""
            x0, y0, r0, xl = vars
            xu, yu, Au, Cu = args

            Al, _, Cl = line_coefs(func=Fu, x0=xl)

            return [abs(Au * x0 + (-1) * y0 + Cu) / sqrt(Au ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xu - x0) ** 2 + (yu - y0) ** 2) - r0 ** 2,  # до кривой корыта
                    abs(Al * x0 + (-1) * y0 + Cl) / sqrt(Al ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xl - x0) ** 2 + (Fu(xl) - y0) ** 2) - r0 ** 2]  # до кривой спинки

        self.d, self.xd, self.yd = list(), list(), list()

        warnings.filterwarnings('error')
        for xu, yu, a_u, c_u in tqdm(zip(x, Fd(x), Au, Cu), desc='Channel calculation', total=len(x)):
            try:
                res = fsolve(equations, [xu, yu, self.__t_b / 2, xu], args=(xu, yu, a_u, c_u))
            except Exception:
                continue

            if xgmin <= res[0] <= xgmax and xgmin <= res[3] <= xgmax and res[2] <= self.__t_b / 2:
                self.d.append(res[2] * 2)
                self.xd.append(res[0])
                self.yd.append(res[1])
        warnings.filterwarnings('default')

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
        plt.plot([min(self.coords['u']['x'] + self.coords['l']['x'])] * 2, [-1, 1],
                 [max(self.coords['u']['x'] + self.coords['l']['x'])] * 2, [-1, 1],
                 ls='-', color='black')  # границы решетки
        for i in range(len(self.d)):
            plt.plot(list(self.d[i] / 2 * cos(alpha) + self.xd[i] for alpha in linspace(0, 2 * pi, 360)),
                     list(self.d[i] / 2 * sin(alpha) + self.yd[i] for alpha in linspace(0, 2 * pi, 360)),
                     ls='solid', color='green')
        plt.plot(self.xd, self.yd, ls='dashdot', color='orange',
                 label=f'gamma = {self.__gamma:.4f} [rad] = {degrees(self.__gamma):.4f} [deg]')
        plt.plot(self.coords['u']['x'], list(y - self.__t_b / 2 for y in self.coords['u']['y']),
                 ls='-', color='black', label=f't/b = {self.__t_b:.4f}')
        plt.plot(self.coords['l']['x'], list(y - self.__t_b / 2 for y in self.coords['l']['y']),
                 ls='-', color='black')
        plt.plot(self.coords['u']['x'], list(y + self.__t_b / 2 for y in self.coords['u']['y']),
                 ls='-', color='black')
        plt.plot(self.coords['l']['x'], list(y + self.__t_b / 2 for y in self.coords['l']['y']),
                 ls='-', color='black')
        plt.legend(fontsize=12)

        fg.add_subplot(gs[0, 1])  # позиция графика
        plt.title('Channel')
        plt.grid(True)
        plt.axis('square')
        plt.xlim([0, ceil(max(self.r))])
        plt.ylim([0, ceil(self.__t_b)])
        plt.plot(self.r, self.d, ls='-', color='green', label='channel')
        plt.plot([0, ceil(max(self.r))], [0, 0], ls='-', color=(0, 0, 0), linewidth=1.5)
        plt.plot(list((self.r[i] + self.r[i + 1]) / 2 for i in range(len(self.r) - 1)),
                 list((self.d[i + 1] - self.d[i]) / (self.r[i + 1] - self.r[i]) for i in range(len(self.r) - 1)),
                 ls='-', color='red', label='d2f/dx2')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


def test() -> None:
    """Тестирование"""
    # print(Disk.version())

    Airfoil.rnd = 4

    airfoils, grates = list(), list()

    if 1:
        airfoils.append(Airfoil('BMSTU', 30))

        airfoils[-1].xg_b = 0.4
        airfoils[-1].r_inlet_b, airfoils[-1].r_outlet_b = 0.06, 0.03
        airfoils[-1].g_ = 0.5
        airfoils[-1].g_inlet, airfoils[-1].g_outlet = radians(20), radians(10)
        airfoils[-1].e = radians(110)
        airfoils[-1].t_b = 1 / 1.698
        airfoils[-1].gamma = radians(46.23)

    if 1:
        airfoils.append(Airfoil('NACA', 40))

        airfoils[-1].c_b = 0.24
        airfoils[-1].f_b = 0.05
        airfoils[-1].xf_b = 0.3
        airfoils[-1].t_b = 1 / 1.698
        airfoils[-1].gamma = radians(46.23)

    if 1:
        airfoils.append(Airfoil('MYNK', 20))

        airfoils[-1].h = 0.1
        airfoils[-1].t_b = 1 / 1.698
        airfoils[-1].gamma = radians(46.23)

    if 1:
        airfoils.append(Airfoil('PARSEC', 30))

        airfoils[-1].r_inlet_b = 0.01
        airfoils[-1].f_b_u, airfoils[-1].f_b_l = (0.35, 0.055), (0.45, -0.006)
        airfoils[-1].d2y_dx2_u, airfoils[-1].d2y_dx2_l = -0.35, -0.2
        airfoils[-1].theta_outlet_u, airfoils[-1].theta_outlet_l = radians(-6), radians(0.05)
        airfoils[-1].t_b = 1 / 1.698
        airfoils[-1].gamma = radians(46.23)

    if 1:
        airfoils.append(Airfoil('BEZIER', 30))

        airfoils[-1].u = ((0.0, 0.0), (0.05, 0.100), (0.35, 0.200), (1.0, 0.0))
        airfoils[-1].l = ((0.0, 0.0), (0.05, -0.10), (0.35, -0.05), (0.5, 0.0), (1.0, 0.0))
        airfoils[-1].t_b = 1 / 1.698
        airfoils[-1].gamma = radians(46.23)

    if 1:
        airfoils.append(Airfoil('MANUAL', 30))

        airfoils[-1].u = ((0.0, 0.0), (0.10, 0.100), (0.35, 0.150), (0.5, 0.15), (1.0, 0.0))
        airfoils[-1].l = ((0.0, 0.0), (0.05, -0.05), (0.35, -0.05), (0.5, 0.0), (1.0, 0.0))
        airfoils[-1].deg = 3
        airfoils[-1].t_b = 1 / 1.698
        airfoils[-1].gamma = radians(46.23)

    for airfoil in airfoils:
        airfoil.solve()
        airfoil.show()
        airfoil.show_grate()

        print(airfoil.to_dataframe(bears='pandas'))
        print(airfoil.to_dataframe(bears='polars'))
        print(airfoil.to_array())

        print(Fore.MAGENTA + 'airfoil properties:' + Fore.RESET)
        for k, v in airfoil.properties.items(): print(f'{k}: {v}')

        print(Fore.MAGENTA + 'grate properties:' + Fore.RESET)
        for k, v in airfoil.properties.items(): print(f'{array(k)}: {v}')

        airfoil.export()


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
