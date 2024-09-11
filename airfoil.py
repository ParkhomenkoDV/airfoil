"""
Список литературы:

[1] = Теория и проектирование газовой турбины: учебное пособие /
В.Е. Михальцев, В.Д. Моляков; под ред. А.Ю. Вараксина. -
Москва: Издательство МГТУ им. Н.Э. Баумана, 2020. - 230, [2] с.: ил.
"""

import sys
import warnings
from copy import deepcopy
from typing import Tuple

from tqdm import tqdm
from colorama import Fore

import numpy as np
from numpy import array, arange, linspace, zeros, full
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
    __methods = {
        'BMSTU': {
            'description': '',
            'aliases': ('BMSTU', 'МГТУ', 'МВТУ', 'МИХАЛЬЦЕВ'),
            'attributes': {
                'rotation_angle': {
                    'description': 'угол поворота потока',
                    'unit': '[rad]',
                    'bounds': f'(0, {radians(180)}]',
                    'type': (int, float, np.number)},
                'relative_inlet_radius': {
                    'description': 'относительный радиус входной кромки',
                    'unit': '[]',
                    'bounds': '(0, 1)',
                    'type': (float, np.floating)},
                'relative_outlet_radius': {
                    'description': 'относительный радиус выходной кромки',
                    'unit': '[]',
                    'bounds': '(0, 1)',
                    'type': (float, np.floating)},
                'inlet_angle': {
                    'description': 'угол раскрытия входной кромки',
                    'unit': '[rad]',
                    'bounds': f'[0, {radians(180)})',
                    'type': (int, float, np.number)},
                'outlet_angle': {
                    'description': 'угол раскрытия выходной кромки',
                    'unit': '[rad]',
                    'bounds': f'[0, {radians(180)})',
                    'type': (int, float, np.number)},
                'x_ray_cross': {
                    'description': 'относительная координата х пересечения входного и выходного лучей',
                    'unit': '[]',
                    'bounds': '(0, 1)',
                    'type': (float, np.floating)},
                'upper_proximity': {
                    'description': 'степень приближенности к спинке',
                    'unit': '[]',
                    'bounds': '[0, 1]',
                    'type': (int, float, np.number)}}},
        'NACA': {'description': '',
                 'aliases': ('NACA', 'N.A.C.A.'),
                 'attributes': {
                     'relative_thickness': {
                         'description': 'максимальная относительная толщина',
                         'unit': '[]',
                         'bounds': '[0, 1)',  # TODO 0 не работает - исправить код, а не границу!
                         'type': (int, float, np.number)},
                     'x_relative_camber': {
                         'description': 'относительна координата х максимальной выпуклости',
                         'unit': '[]',
                         'bounds': '[0, 1]',
                         'type': (int, float, np.number)},
                     'relative_camber': {
                         'description': 'относительная максимальная выпуклость',
                         'unit': '[]',
                         'bounds': '[0, 1)',
                         'type': (int, float, np.number)},
                     'closed': {
                         'description': 'замкнутость профиля',
                         'unit': '[]',
                         'bounds': '{False, True}',
                         'type': (bool,)}}},
        'MYNK': {'description': '',
                 'aliases': ('MYNK', 'МУНК'),
                 'attributes': {
                     'mynk_coefficient': {
                         'description': 'коэффициент Мунка',
                         'unit': '[]',
                         'bounds': (0, 1),
                         'type': ()}}},
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

    @classmethod
    def __version__(cls):
        version = '3.0'
        print('help')
        print('Продувка')

    # TODO
    @classmethod
    def help(cls):
        """Помощь при работе с классом Airfoil и его объектами"""
        print('Airfoil.rnd = 4  # количество значащих цифр')
        print('airfoil = Airfoil(method, discreteness, relative_step, gamma)  # создание объекта')
        print('где:')
        print('discreteness:  int >= 3      # количество дискретных точек')
        print('relative_step: int > 0       # относительный шаг')
        print('gamma:         float < pi/2  # угол установки профиля')
        print('methods:  # методы построения аэродинамического профиля')
        for method in Airfoil.__methods:
            print(method)
            print('\t' + f'description: {Airfoil.__methods[method]["description"]}')
            print('\t' + f'aliases: {Airfoil.__methods[method]["aliases"]}')
            print('\t' + f'attributes:')
            for attribute in Airfoil.__methods[method]["attributes"]:
                print('\t\t' + f'{attribute}')
                for key, value in Airfoil.__methods[method]["attributes"][attribute].items():
                    print('\t\t\t' + f'{key}: {value}')
        print('airfoil.show()')
        print('airfoil.properties')
        print('airfoil.channel')
        print('airfoil.to_dataframe()')
        print('airfoil.export()')

    @property
    @classmethod
    def rnd(cls) -> int:
        return cls.rnd

    @rnd.setter
    @classmethod
    def rnd(cls, value):
        assert isinstance(value, int)
        Airfoil.rnd = value

    def validate(self, **kwargs) -> None:
        """Проверка верности ввода атрибутов профиля"""

        method = kwargs.pop('method', None)
        if method is not None:
            assert isinstance(method, str)
            method = method.strip().upper()
            assert any(method in value['aliases'] for value in Airfoil.__methods.values())

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
            for attr in Airfoil.__methods[self.__method]['attributes']:
                assert hasattr(self, attr), f'not hasattr({attr})'
                assert isinstance(getattr(self, attr), Airfoil.__methods[self.__method]['attributes'][attr]['type']), \
                    f'type({attr}) not in {Airfoil.__methods[self.__method]["attributes"][attr]["type"]}'

                bounds = Airfoil.__methods[self.__method]['attributes'][attr]['bounds']
                if bounds[0] == '{':  # множества
                    union, exclusion = bounds.split(' \ ') if ' \ ' in bounds else bounds, None

                    union = union[1:-1]  # удаление '{' и '}'
                    if '"' in union:  # строковое множество
                        assert any(getattr(self, attr) == el for el in union.split(', '))
                    elif 'True' in union or 'False' in union:  # булево множество
                        assert any(getattr(self, attr) == bool(el) for el in union.split(', '))
                    else:
                        assert any(getattr(self, attr) == float(el) for el in union.split(', '))

                    if exclusion is not None:
                        exclusion = exclusion[1:-1]  # удаление '{' и '}'
                        if '"' in exclusion:  # строковое множество
                            assert all(getattr(self, attr) != el for el in exclusion.split(', '))
                        elif 'True' in union or 'False' in union:  # булево множество
                            assert all(getattr(self, attr) != bool(el) for el in union.split(', '))
                        else:
                            assert all(getattr(self, attr) != float(el) for el in exclusion.split(', '))
                else:  # интервалы
                    l, u = bounds.split(', ')
                    if l[1] != '_':  # есть нижняя граница
                        if l[0] == '(':
                            assert float(l[1:]) < getattr(self, attr), f'attribute {attr} > {float(l[1:])}'
                        elif l[0] == '[':
                            assert float(l[1:]) <= getattr(self, attr), f'attribute {attr} >= {float(l[1:])}'
                    if u[-2] != '_':  # есть верхняя граница
                        if u[-1] == ')':
                            assert getattr(self, attr) < float(u[:-1]), f'attribute {attr} < {float(u[:-1])}'
                        elif u[-1] == ']':
                            assert getattr(self, attr) <= float(u[:-1]), f'attribute {attr} <= {float(u[:-1])}'

            if self.__method in Airfoil.__methods['MYNK']['aliases']:
                assert hasattr(self, 'mynk_coefficient')
                assert isinstance(self.mynk_coefficient, (int, float))
                assert 0 <= self.mynk_coefficient <= 1

            elif self.__method in Airfoil.__methods['PARSEC']['aliases']:
                # относ. радиус входной кромки
                assert hasattr(self, 'r_inlet_b')
                assert isinstance(self.relative_inlet_radius, (int, float))
                assert 0 <= self.relative_inlet_radius <= 1

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

            elif self.__method in Airfoil.__methods['BEZIER']['aliases']:
                assert hasattr(self, 'u') and hasattr(self, 'l')
                validate_points(self.u)
                validate_points(self.l)

            elif self.__method in Airfoil.__methods['MANUAL']['aliases']:
                assert hasattr(self, 'u') and hasattr(self, 'l')
                validate_points(self.u)
                validate_points(self.l)
                assert hasattr(self, 'deg') and isinstance(self.deg, int) and 0 <= self.deg <= 3

    def __init__(self, method: str, discreteness: int = __discreteness,
                 relative_step: float | int = __relative_step, gamma: float | int = __gamma, **attributes):
        self.validate(method=method, discreteness=discreteness, relative_step=relative_step, gamma=gamma)

        self.__method = method.strip().upper()  # метод построения аэродинамического профиля
        self.__discreteness = discreteness  # количество точек дискретизации

        self.__relative_step = relative_step  # относительный шаг []
        self.__gamma = gamma  # угол установки [рад]

        self.__coordinates = tuple()  # относительные координаты профиля считая против часовой стрелки с выходной кромки
        self.__properties = dict()  # относительные характеристики профиля
        self.__channel = tuple()  # дифузорность/конфузорность решетки

        for attribute, value in attributes.items():
            setattr(self, attribute, value)

    def __str__(self) -> str:
        return self.__method

    def __setattr__(self, key, value):
        """При установке новых атрибутов расчет обнуляется"""
        if not key.startswith('_'): self.__coordinates, self.__properties, self.__channel = tuple(), dict(), tuple()
        object.__setattr__(self, key, value)

    @property
    def method(self) -> str:
        return self.__method

    @method.setter
    def method(self, value: str) -> None:
        self.validate(method=value)
        self.__init__(value)  # снос предыдущих расчетов для нового метода

    @method.deleter
    def method(self) -> None:
        raise

    @property
    def discreteness(self) -> int:
        return self.__discreteness

    @discreteness.setter
    def discreteness(self, value) -> None:
        self.validate(discreteness=value)
        self.__init__(method=self.method, discreteness=value)

    @discreteness.deleter
    def discreteness(self) -> None:
        raise

    @property
    def relative_step(self) -> float | int | np.number:
        return self.__relative_step

    @relative_step.setter
    def relative_step(self, value):
        self.validate(relative_step=value)
        self.__relative_step = value
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
    def coordinates(self) -> tuple[tuple[float, float], ...]:
        if len(self.__coordinates) == 0: self.__calculate()
        return self.__coordinates

    # TODO
    def input(self):
        """Динамический ввод с защитой от дураков"""
        pass

    def __bmstu(self) -> tuple[tuple[float, float], ...]:
        # tan угла входа и выхода потока
        k_inlet = 1 / (2 * self.x_ray_cross / (self.x_ray_cross - 1) * tan(self.rotation_angle))
        k_outlet = 1 / (2 * tan(self.rotation_angle))
        if tan(self.rotation_angle) * self.rotation_angle > 0:
            k_inlet *= ((self.x_ray_cross / (self.x_ray_cross - 1) - 1) -
                        sqrt((self.x_ray_cross / (self.x_ray_cross - 1) - 1) ** 2 -
                             4 * (self.x_ray_cross / (self.x_ray_cross - 1) * tan(self.rotation_angle) ** 2)))
            k_outlet *= ((self.x_ray_cross / (self.x_ray_cross - 1) - 1) -
                         sqrt((self.x_ray_cross / (self.x_ray_cross - 1) - 1) ** 2 -
                              4 * (self.x_ray_cross / (self.x_ray_cross - 1) * tan(self.rotation_angle) ** 2)))
        else:
            k_inlet *= ((self.x_ray_cross / (self.x_ray_cross - 1) - 1) +
                        sqrt((self.x_ray_cross / (self.x_ray_cross - 1) - 1) ** 2 -
                             4 * (self.x_ray_cross / (self.x_ray_cross - 1) * tan(self.rotation_angle) ** 2)))
            k_outlet *= ((self.x_ray_cross / (self.x_ray_cross - 1) - 1) +
                         sqrt((self.x_ray_cross / (self.x_ray_cross - 1) - 1) ** 2 -
                              4 * (self.x_ray_cross / (self.x_ray_cross - 1) * tan(self.rotation_angle) ** 2)))

        # углы входа и выхода профиля
        if self.rotation_angle > 0:
            g_u_inlet, g_d_inlet = ((1 - self.upper_proximity) *
                                    self.inlet_angle, self.upper_proximity * self.inlet_angle)
            g_u_outlet, g_d_outlet = ((1 - self.upper_proximity) *
                                      self.outlet_angle, self.upper_proximity * self.outlet_angle)
        else:
            g_u_inlet, g_d_inlet = self.upper_proximity * self.inlet_angle, (
                    1 - self.upper_proximity) * self.inlet_angle,
            g_u_outlet, g_d_outlet = self.upper_proximity * self.outlet_angle, (
                    1 - self.upper_proximity) * self.outlet_angle

        # относительные радиусы входной и выходной кромок
        self.__relative_inlet_radius = self.relative_inlet_radius
        self.__relative_outlet_radius = self.relative_outlet_radius
        # положения центров окружностей входной и выходной кромок
        O_inlet = self.__relative_inlet_radius, k_inlet * self.__relative_inlet_radius
        O_outlet = 1 - self.__relative_outlet_radius, -k_outlet * self.__relative_outlet_radius

        # точки пересечения линий спинки и корыта
        xcl_u, ycl_u = COOR(tan(atan(k_inlet) + g_u_inlet),
                            sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * self.__relative_inlet_radius -
                            (tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1],
                            tan(atan(k_outlet) - g_u_outlet),
                            sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * self.__relative_outlet_radius -
                            (tan(atan(k_outlet) - g_u_outlet)) * O_outlet[0] - (-1) * O_outlet[1])

        xcl_d, ycl_d = COOR(tan(atan(k_inlet) - g_d_inlet),
                            -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * self.__relative_inlet_radius -
                            (tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1],
                            tan(atan(k_outlet) + g_d_outlet),
                            -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * self.__relative_outlet_radius -
                            (tan(atan(k_outlet) + g_d_outlet)) * O_outlet[0] - (-1) * O_outlet[1])

        # точки пересечения окружностей со спинкой и корытом
        xclc_i_u, yclc_i_u = COOR(tan(atan(k_inlet) + g_u_inlet),
                                  sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * self.__relative_inlet_radius
                                  - (tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1],
                                  -1 / (tan(atan(k_inlet) + g_u_inlet)),
                                  -(-1 / tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1])

        xclc_i_d, yclc_i_d = COOR(tan(atan(k_inlet) - g_d_inlet),
                                  -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * self.__relative_inlet_radius
                                  - (tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1],
                                  -1 / (tan(atan(k_inlet) - g_d_inlet)),
                                  -(-1 / tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1])

        xclc_e_u, yclc_e_u = COOR(tan(atan(k_outlet) - g_u_outlet),
                                  sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * self.__relative_outlet_radius
                                  - tan(atan(k_outlet) - g_u_outlet) * O_outlet[0] - (-1) * O_outlet[1],
                                  -1 / tan(atan(k_outlet) - g_u_outlet),
                                  -(-1 / tan(atan(k_outlet) - g_u_outlet)) * O_outlet[0] - (-1) * O_outlet[1])

        xclc_e_d, yclc_e_d = COOR(tan(atan(k_outlet) + g_d_outlet),
                                  -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * self.__relative_outlet_radius
                                  - tan(atan(k_outlet) + g_d_outlet) * O_outlet[0] - (-1) * O_outlet[1],
                                  -1 / tan(atan(k_outlet) + g_d_outlet),
                                  -(-1 / tan(atan(k_outlet) + g_d_outlet)) * O_outlet[0] - (-1) * O_outlet[1])

        x, y = list(), list()

        # окружность выходной кромки спинки
        an = angle(points=((1, O_outlet[1]), O_outlet, (xclc_e_u, yclc_e_u)))
        if O_outlet[0] > xclc_e_u: an = pi - an
        # уменьшение угла для предотвращения дублирования координат
        angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
        x += (1 - self.__relative_outlet_radius * (1 - cos(angles))).tolist()
        y += (O_outlet[1] + self.__relative_outlet_radius * sin(angles)).tolist()

        # спинка
        xu, yu = bernstein_curve(((xclc_e_u, yclc_e_u), (xcl_u, ycl_u), (xclc_i_u, yclc_i_u)),
                                 N=self.__discreteness).T.tolist()
        x += xu
        y += yu

        # точки входной окружности кромки по спинке
        an = angle(points=((0, O_inlet[1]), O_inlet, (xclc_i_u, yclc_i_u)))
        if xclc_i_u > O_inlet[0]: an = pi - an
        # уменьшение угла для предотвращения дублирования координат
        angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
        x += (self.__relative_inlet_radius * (1 - cos(angles))).tolist()[::-1]
        y += (O_inlet[1] + self.__relative_inlet_radius * sin(angles)).tolist()[::-1]

        x.pop(), y.pop()  # удаление дубликата входной точки

        # окружность входной кромки корыта
        an = angle(points=((0, O_inlet[1]), O_inlet, (xclc_i_d, yclc_i_d)))
        if xclc_i_d > O_inlet[0]: an = pi - an
        # уменьшение угла для предотвращения дублирования координат
        angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
        x += (self.__relative_inlet_radius * (1 - cos(angles))).tolist()
        y += (O_inlet[1] - self.__relative_inlet_radius * sin(angles)).tolist()

        # корыто
        xd, yd = bernstein_curve(((xclc_i_d, yclc_i_d), (xcl_d, ycl_d), (xclc_e_d, yclc_e_d)),
                                 N=self.__discreteness).T.tolist()
        x += xd
        y += yd

        # точки выходной окружности кромки по корыту
        an = angle(points=((1, O_outlet[1]), O_outlet, (xclc_e_d, yclc_e_d)))
        if O_outlet[0] > xclc_e_d: an = pi - an
        # уменьшение угла для предотвращения дублирования координат
        angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
        x += (1 - self.__relative_outlet_radius * (1 - cos(angles))).tolist()[::-1]
        y += (O_outlet[1] - self.__relative_outlet_radius * sin(angles)).tolist()[::-1]

        return tuple(((x, y) for x, y in zip(x, y)))

    def __naca(self) -> tuple[tuple[float, float], ...]:
        i = arange(self.__discreteness)  # массив индексов
        betta = i * pi / (2 * (self.__discreteness - 1))
        x = 1 - cos(betta)

        mask = (0 <= x) & (x <= self.x_relative_camber)

        yf = np.full_like(i, self.relative_camber, dtype=np.float64)
        yf[mask] *= self.x_relative_camber ** (-2) * (2 * self.x_relative_camber * x[mask] - x[mask] ** 2)
        yf[~mask] *= (1 - self.x_relative_camber) ** (-2) * (
                1 - 2 * self.x_relative_camber + 2 * self.x_relative_camber * x[~mask] - x[~mask] ** 2)

        gradYf = 2 * self.relative_camber

        a = array((0.2969, -0.126, -0.3516, 0.2843, -0.1036 if self.closed else -0.1015), dtype='float64')

        yc = self.relative_thickness / 0.2 * np.dot(a, np.column_stack((sqrt(x), x, x ** 2, x ** 3, x ** 4)).T)

        tetta = atan(gradYf)

        sin_tetta, cos_tetta = sin(tetta), cos(tetta)  # предварительный расчет для ускорения работы

        X = np.hstack(((x - yc * sin_tetta)[::-1], (x + yc * sin_tetta)[1::]))  # revers против часовой стрелки
        Y = np.hstack(((yf + yc * cos_tetta)[::-1], (yf - yc * cos_tetta)[1::]))  # удаление дубликатной входной точки

        x_min, x_max = X.min(), X.max()
        scale = abs(x_max - x_min)

        coordinates = self.__transform(tuple(((x, y) for x, y in zip(X, Y))), x0=x_min, scale=(1 / scale))

        self.__relative_outlet_radius = 0 if self.closed else abs(coordinates[0][1] - coordinates[-1][1])

        return coordinates

    def __mynk_coordinates(self, param, x) -> tuple:
        part1, part2 = 0.25 * (-x - 17 * x ** 2 - 6 * x ** 3), x ** 0.87 * (1 - x) ** 0.56
        return param * (part1 + part2), param * (part1 - part2)

    def __mynk(self) -> tuple[tuple[float, float], ...]:
        coordinates = {'u': {'x': list(), 'y': list()}, 'l': {'x': list(), 'y': list()}}  # результат

        x = linspace(0, 1, self.__discreteness)
        coordinates['u']['x'], coordinates['l']['x'] = x, x
        coordinates['u']['y'], coordinates['l']['y'] = self.__mynk_coordinates(self.mynk_coefficient, x)

        angle = atan((coordinates['u']['y'][-1] - coordinates['u']['y'][0]) / (1 - 0))
        scale = dist((coordinates['u']['x'][0], coordinates['u']['y'][0]),
                     (coordinates['u']['x'][-1], coordinates['u']['y'][-1]))

        ux, uy = [nan] * len(coordinates['u']['x']), [nan] * len(coordinates['u']['x'])
        lx, ly = [nan] * len(coordinates['l']['x']), [nan] * len(coordinates['l']['x'])

        for i, _ in enumerate(coordinates['u']['x']):
            ux[i], uy[i] = Axis.transform(coordinates['u']['x'][i], coordinates['u']['y'][i],
                                          x0=0, y0=0, angle=angle, scale=1 / scale)
        for i, _ in enumerate(coordinates['l']['x']):
            lx[i], ly[i] = Axis.transform(coordinates['l']['x'][i], coordinates['l']['y'][i],
                                          x0=0, y0=0, angle=angle, scale=1 / scale)

        # отсечка значений спинки корыту и наоборот
        Xu = (ux[ux.index(min(ux)):] + list(reversed(lx[lx.index(max(lx)):len(lx) - 1])))
        Yu = (uy[ux.index(min(ux)):] + list(reversed(lx[lx.index(max(lx)):len(lx) - 1])))
        Xl = list(reversed(ux[1:ux.index(min(ux)) + 1])) + lx[:lx.index(max(lx)) + 1]
        Yl = list(reversed(uy[1:ux.index(min(ux)) + 1])) + ly[:lx.index(max(lx)) + 1]

        coordinates = {'u': {'x': Xu, 'y': Yu}, 'l': {'x': Xl, 'y': Yl}}

        return coordinates

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

    def __parsec(self) -> dict[str:dict[str:list]]:
        """
        Generate and plot the contour of an airfoil using the PARSEC parameterization
        H. Sobieczky, *'Parametric airfoils and wings'* in *Notes on Numerical Fluid Mechanics*, Vol. 68, pp 71-88]
        (www.as.dlr.de/hs/h-pdf/H141.pdf)
        Repository & documentation: http://github.com/dqsis/parsec-airfoils
        """

        # поверхностные коэффициенты давления спинки и корыта
        cf_u = self.__parsec_coefficients('u', self.relative_inlet_radius, self.f_b_u, self.d2y_dx2_u, (1, 0),
                                          self.theta_outlet_u)
        cf_l = self.__parsec_coefficients('l', self.relative_inlet_radius, self.f_b_l, self.d2y_dx2_l, (1, 0),
                                          self.theta_outlet_l)

        self.coordinates['u']['x'] = linspace(0, 1, self.__discreteness)
        self.coordinates['u']['y'] = sum([cf_u[i] * self.coordinates['u']['x'] ** (i + 0.5) for i in range(6)])
        self.coordinates['l']['x'] = linspace(1, 0, self.__discreteness)
        self.coordinates['l']['y'] = sum([cf_l[i] * self.coordinates['l']['x'] ** (i + 0.5) for i in range(6)])
        self.coordinates['l']['x'], self.coordinates['l']['y'] = self.coordinates['l']['x'][::-1], \
            self.coordinates['l']['y'][::-1]

        self.__O_inlet, self.relative_inlet_radius = (0, 0), 0
        self.__O_outlet, self.relative_outlet_radius = (1, 0), 0

    def __bezier(self) -> dict[str:dict[str:list]]:
        if not any(p[0] == 0 for p in self.u): self.u = list(self.u) + [(0, 0)]
        if not any(p[0] == 1 for p in self.u): self.u = list(self.u) + [(1, 0)]

        self.coordinates['u']['x'], self.coordinates['u']['y'] = bernstein_curve(self.u, N=self.__N).T
        self.coordinates['l']['x'], self.coordinates['l']['y'] = bernstein_curve(self.l, N=self.__N).T

        self.__O_inlet, self.relative_inlet_radius = (0, 0), 0
        self.__O_outlet, self.relative_outlet_radius = (1, 0), 0

    def __manual(self) -> dict[str:dict[str:list]]:
        if not any(p[0] == 0 for p in self.u): self.u = list(self.u) + [(0, 0)]
        if not any(p[0] == 1 for p in self.u): self.u = list(self.u) + [(1, 0)]

        x = linspace(0, 1, self.__discreteness)
        self.coordinates['u']['x'], self.coordinates['u']['y'] = x, interpolate.interp1d([p[0] for p in self.u],
                                                                                         [p[1] for p in self.u],
                                                                                         kind=self.deg)(x)
        self.coordinates['l']['x'], self.coordinates['l']['y'] = x, interpolate.interp1d([p[0] for p in self.l],
                                                                                         [p[1] for p in self.l],
                                                                                         kind=self.deg)(x)

        self.__O_inlet, self.relative_inlet_radius = (0, 0), 0
        self.__O_outlet, self.relative_outlet_radius = (1, 0), 0

    @timeit()
    def __calculate(self) -> tuple[tuple[float, float], ...]:
        self.validate()

        if self.method in Airfoil.__methods['NACA']['aliases']:
            self.__coordinates0 = self.__naca()
        elif self.method in Airfoil.__methods['BMSTU']['aliases']:
            self.__coordinates0 = self.__bmstu()
        elif self.method in Airfoil.__methods['MYNK']['aliases']:
            self.__coordinates0 = self.__mynk()
        elif self.method in Airfoil.__methods['PARSEC']['aliases']:
            self.__coordinates0 = self.__parsec()
        elif self.method in Airfoil.__methods['BEZIER']['aliases']:
            self.__coordinates0 = self.__bezier()
        elif self.method in Airfoil.__methods['MANUAL']['aliases']:
            self.__coordinates0 = self.__manual()
        else:
            print(Fore.RED + f'No such method {self.method}! Use Airfoil.help' + Fore.RESET)

        self.__coordinates = self.__transform(self.__coordinates0, angle=self.__gamma, inplace=False)  # поворот
        coordinates = array(self.__coordinates, dtype='float64').T
        x_min, x_max = coordinates[0].min(), coordinates[0].max()
        scale = abs(x_max - x_min)
        self.__coordinates = self.__transform(self.__coordinates, x0=x_min, scale=(1 / scale), inplace=False)  # нормал
        return self.__coordinates

    def __transform(self, coordinates: tuple[tuple[float, float], ...],
                    x0=0.0, y0=0.0, angle=0.0, scale=1.0, inplace: bool = False) -> tuple[tuple[float, float], ...]:
        """Перенос-поворот кривых спинки и корыта профиля"""

        new_coordinates = list()
        for x, y in coordinates:
            point = Axis.transform(x, y, x0=x0, y0=y0, angle=angle, scale=scale)
            new_coordinates.append((float(point[0]), float(point[1])))
        new_coordinates = tuple(new_coordinates)

        if inplace: self.__coordinates = new_coordinates

        return new_coordinates

    @staticmethod
    def to_upper_lower(coordinates: tuple[tuple[float, float], ...]) -> dict[str:tuple[tuple[float, float], ...]]:
        """Разделение координат на спинку и корыто"""
        X, Y = array(coordinates).T
        argmin, argmax = np.argmin(X), np.argmax(X)
        upper, lower = list(), list()
        if argmin < argmax:
            for x, y in zip(X[argmax:-1:+1], Y[argmax:-1:+1]): upper.append((float(x), float(y)))
            for x, y in zip(X[:argmin + 1:+1], Y[:argmin + 1:+1]): upper.append((float(x), float(y)))
            for x, y in zip(X[argmin:argmax + 1:+1], Y[argmin:argmax + 1:+1]): lower.append((float(x), float(y)))
        else:
            for x, y in zip(X[argmax:argmin + 1:+1], Y[argmax:argmin + 1:+1]): upper.append((float(x), float(y)))
            for x, y in zip(X[argmin:-1:+1], Y[argmin:-1:+1]): lower.append((float(x), float(y)))
            for x, y in zip(X[:argmax + 1:+1], Y[:argmax + 1:+1]): lower.append((float(x), float(y)))
        # if upper[-1][0] != 0: upper.append((0, ?)) # неизвестен y входной кромки
        return {'upper': tuple(upper[::-1]), 'lower': tuple(lower)}

    def __find_circles(self, coordinates: tuple[tuple[float, float], ...], dl: float = 0.01) -> dict[str:dict]:
        """Поиск радиусов окружностей входной и выходной кромок и их центров"""
        # dl < 0.01 нецелесообразен ввиду технологической невозможности
        # dl > 0.01 нецелесообразен ввиду большого шага dx производной

        coordinates = self.to_upper_lower(coordinates)

        Fu = interpolate.interp1d(*(array(coordinates['upper']).T), kind=3, fill_value='extrapolate')
        Fl = interpolate.interp1d(*(array(coordinates['lower']).T), kind=3, fill_value='extrapolate')

        x0, x1 = 0, 1  # координаты x входной и выходной окружности
        y0, y1 = Fu(x0), Fu(x1)  # координаты y входной и выходной окружности

        x0u = x0 + dl * tan2cos(derivative(Fu, x0, method='forward', dx=dl))
        x1u = x1 - dl * tan2cos(derivative(Fu, x1, method='backward', dx=dl))
        y0u, y1u = Fu(x0u), Fu(x1u)
        x0l = x0 + dl * tan2cos(derivative(Fl, x0, method='forward', dx=dl))
        x1l = x1 - dl * tan2cos(derivative(Fl, x1, method='backward', dx=dl))
        y0l, y1l = Fl(x0l), Fl(x1l)

        A0u, B0u, C0u = line_coefs(p1=(x0, y0), p2=(x0u, y0u))
        A0l, B0l, C0l = line_coefs(p1=(x0, y0), p2=(x0l, y0l))
        A1u, B1u, C1u = line_coefs(p1=(x1, y1), p2=(x1u, y1u))
        A1l, B1l, C1l = line_coefs(p1=(x1, y1), p2=(x1l, y1l))

        # коэффициент A для перпендикуляров
        AA0u, AA0l = -1 / A0u if A0u != 0 else -inf, -1 / A0l if A0l != 0 else -inf
        AA1u, AA1l = -1 / A1u if A1u != 0 else -inf, -1 / A1l if A1l != 0 else -inf
        # коэффициент С для перпендикуляров
        CC0u, CC0l = np.mean((y0, y0u)) - AA0u * np.mean((x0, x0u)), np.mean((y0, y0l)) - AA0l * np.mean((x0, x0l))
        CC1u, CC1l = np.mean((y1, y1u)) - AA1u * np.mean((x1, x1u)), np.mean((y1, y1l)) - AA1l * np.mean((x1, x1l))

        # центры входной и выходной окружностей
        O_inlet, O_outlet = COOR(AA0u, CC0u, AA0l, CC0l), COOR(AA1u, CC1u, AA1l, CC1l)
        if not (0.0 <= O_inlet[0] <= 0.5) or not (Fl(O_inlet[0]) <= O_inlet[1] <= Fu(O_inlet[0])):
            O_inlet = (nan, nan)
        if not (0.5 <= O_outlet[0] <= 1.0) or not (Fl(O_outlet[0]) <= O_outlet[1] <= Fu(O_outlet[0])):
            O_outlet = (nan, nan)

        if not hasattr(self, '_Airfoil__relative_inlet_radius'): self.__relative_inlet_radius = abs(O_inlet[0] - x0)
        if not hasattr(self, '_Airfoil__relative_outlet_radius'): self.__relative_outlet_radius = abs(O_outlet[0] - x1)

        return {'inlet': {'point': O_inlet, 'radius': self.__relative_inlet_radius},
                'outlet': {'point': O_outlet, 'radius': self.__relative_outlet_radius}}

    def show(self, amount: int = 2, figsize=(12, 10), savefig=False):
        """Построение профиля"""
        assert isinstance(amount, int) and 1 <= amount  # количество профилей

        fg = plt.figure(figsize=figsize)
        gs = fg.add_gridspec(nrows=2, ncols=3)

        fg.add_subplot(gs[0, 0])
        plt.title('Initial data')
        plt.axis('off')
        plt.plot([], label=f'method = {self.method}')
        plt.plot([], label=f'discreteness = {self.__discreteness}')
        plt.plot([], label=f'relative_step = {self.__relative_step:.{Airfoil.rnd}f} []')
        plt.plot([],
                 label=f'gamma = {self.__gamma:.{Airfoil.rnd}f} [rad] = {degrees(self.__gamma):.{Airfoil.rnd}f} [deg]')
        for key, value in self.__dict__.items():
            if not key.startswith('_') and type(value) in (int, float, np.number):
                plt.plot([], label=f'{key} = {rounding(value, self.rnd)}')
        plt.legend(loc='upper center')

        fg.add_subplot(gs[1, 0])
        plt.title('Properties')
        plt.axis('off')
        for key, value in self.properties.items(): plt.plot([], label=f'{key} = {rounding(value, self.rnd)}')
        plt.legend(loc='upper center')

        coordinates = self.to_upper_lower(self.__coordinates0)
        print(coordinates['upper'])
        print(coordinates['lower'])

        fg.add_subplot(gs[0, 1])
        plt.title('Airfoil structure')
        plt.grid(True)  # сетка
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.plot(*(array(coordinates['upper']).T), ls='solid', color='blue', linewidth=2)
        plt.plot(*(array(coordinates['lower']).T), ls='solid', color='red', linewidth=2)
        alpha = linspace(0, 2 * pi, 360)
        circles = self.__find_circles(self.__coordinates0)
        x_inlet = self.__relative_inlet_radius * cos(alpha) + circles['inlet']['point'][0]
        y_inlet = self.__relative_inlet_radius * sin(alpha) + circles['inlet']['point'][1]
        x_outlet = self.__relative_outlet_radius * cos(alpha) + circles['outlet']['point'][0]
        y_outlet = self.__relative_outlet_radius * sin(alpha) + circles['outlet']['point'][1]
        plt.plot(x_inlet, y_inlet, ls='solid', color='black', linewidth=1)
        plt.plot(x_outlet, y_outlet, ls='solid', color='black', linewidth=1)

        x, y, d, r = self.channel.T
        X, Y = array(self.__coordinates).T

        fg.add_subplot(gs[1, 1])
        plt.title('Channel')
        plt.grid(True)
        plt.ylim([-self.__relative_step / 2, self.__relative_step / 2])
        plt.plot(r, d / 2, ls='solid', color='green', label='channel')
        plt.plot(r, -d / 2, ls='solid', color='green')
        plt.plot([0, max(r)], [0, 0], ls='dashdot', color='orange', linewidth=1.5)
        plt.plot((r[:-1] + r[1:]) / 2, np.diff(d) / np.diff(r), ls='solid', color='red', label='df/dx')
        plt.axis('equal')  # square
        plt.legend(fontsize=12)

        fg.add_subplot(gs[:, 2])
        plt.title('Lattice')
        plt.grid(True)
        plt.axis('equal')  # xlim не нужен ввиду эквивалентности
        plt.xlim([0, 1])
        plt.plot((0, 0), (np.max(Y), np.min(Y) - (amount - 1) * self.__relative_step),
                 (1, 1), (np.max(Y), np.min(Y) - (amount - 1) * self.__relative_step),
                 ls='solid', color='black')  # границы решетки
        for n in range(amount): plt.plot(X, Y - n * self.__relative_step, ls='solid', color='black', linewidth=2)
        alpha = linspace(0, 2 * pi, 360)
        for i in range(len(d)):
            plt.plot(list(d[i] / 2 * cos(alpha) + x[i]), list(d[i] / 2 * sin(alpha) + y[i]),
                     ls='solid', color='green')
        plt.plot(x, y, ls='dashdot', color='orange')

        if savefig:
            export2(plt, file_path='exports/airfoil', file_name='airfoil', file_extension='png', show_time=False)
        plt.tight_layout()
        plt.show()

    @property
    @timeit()
    def properties(self, epsrel: float = 1e-4) -> dict[str: float]:
        if self.__properties: return self.__properties

        if not hasattr(self, '_Airfoil__relative_inlet_radius'):
            self.__relative_inlet_radius = self.__find_circles(self.coordinates)['inlet']['radius']
        if not hasattr(self, '_Airfoil__relative_outlet_radius'):
            self.__relative_outlet_radius = self.__find_circles(self.coordinates)['outlet']['radius']
        self.__properties['radius_inlet'] = self.__relative_inlet_radius
        self.__properties['radius_outlet'] = self.__relative_outlet_radius

        dct = self.to_upper_lower(self.__coordinates)
        self.__Fu = interpolate.interp1d(*(array(dct['upper']).T), kind=3, fill_value='extrapolate')
        self.__Fl = interpolate.interp1d(*(array(dct['lower']).T), kind=3, fill_value='extrapolate')

        self.__properties['area'] = integrate.dblquad(lambda _, __: 1,
                                                      0, 1, lambda xu: self.__Fl(xu), lambda xl: self.__Fu(xl),
                                                      epsrel=epsrel)[0]
        self.__properties['xc'], self.__properties['c'] = -1.0, 0.0
        self.__properties['xf'], self.__properties['f'] = -1.0, 0.0
        for x in linspace(0, 1, int(ceil(1 / epsrel))):
            if self.__Fu(x) - self.__Fl(x) > self.__properties['c']:
                self.__properties['xc'], self.__properties['c'] = x, self.__Fu(x) - self.__Fl(x)
            if abs((self.__Fu(x) + self.__Fl(x)) / 2) > abs(self.__properties['f']):
                self.__properties['xf'], self.__properties['f'] = x, (self.__Fu(x) + self.__Fl(x)) / 2
        self.__properties['Sx'] = integrate.dblquad(lambda y, _: y,
                                                    0, 1, lambda xu: self.__Fl(xu), lambda xd: self.__Fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Sy'] = integrate.dblquad(lambda _, x: x,
                                                    0, 1, lambda xu: self.__Fl(xu), lambda xd: self.__Fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['x0'] = self.__properties['Sy'] / self.__properties['area'] \
            if self.__properties['area'] != 0 else inf
        self.__properties['y0'] = self.__properties['Sx'] / self.__properties['area'] \
            if self.__properties['area'] != 0 else inf
        self.__properties['Jx'] = integrate.dblquad(lambda y, _: y ** 2,
                                                    0, 1, lambda xu: self.__Fl(xu), lambda xd: self.__Fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Jy'] = integrate.dblquad(lambda _, x: x ** 2,
                                                    0, 1, lambda xu: self.__Fl(xu), lambda xd: self.__Fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Jxy'] = integrate.dblquad(lambda y, x: x * y,
                                                     0, 1, lambda xu: self.__Fl(xu), lambda xd: self.__Fu(xd),
                                                     epsrel=epsrel)[0]
        self.__properties['Jxc'] = self.__properties['Jx'] - self.__properties['area'] * self.__properties['y0'] ** 2
        self.__properties['Jyc'] = self.__properties['Jy'] - self.__properties['area'] * self.__properties['x0'] ** 2
        self.__properties['Jxcyc'] = (self.__properties['Jxy'] -
                                      self.__properties['area'] * self.__properties['x0'] * self.__properties['y0'])
        self.__properties['Jp'] = self.__properties['Jxc'] + self.__properties['Jyc']
        self.__properties['Wp'] = self.__properties['Jp'] / max(
            sqrt((0 - self.__properties['x0']) ** 2 + (0 - self.__properties['y0']) ** 2),
            sqrt((1 - self.__properties['x0']) ** 2 + (0 - self.__properties['y0']) ** 2))
        self.__properties['alpha'] = 0.5 * atan(-2 * self.__properties['Jxcyc'] /
                                                (self.__properties['Jxc'] - self.__properties['Jyc'])) \
            if (self.__properties['Jxc'] - self.__properties['Jyc']) != 0 else -pi / 4
        self.__properties['len_u'] = integrate.quad(lambda x: sqrt(1 + derivative(self.__Fu, x) ** 2),
                                                    0, 1,
                                                    epsrel=epsrel)[0]
        self.__properties['len_l'] = integrate.quad(lambda x: sqrt(1 + derivative(self.__Fl, x) ** 2),
                                                    0, 1,
                                                    epsrel=epsrel)[0]
        return self.__properties

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
    def channel(self) -> np.ndarray:
        """Дифузорность/конфузорность решетки"""
        if len(self.__channel) > 1: return self.__channel

        Fu = lambda x: self.__Fu(x) - self.__relative_step

        xgmin, xgmax = 0 + self.__relative_inlet_radius, 1 - self.__relative_outlet_radius

        step = self.__properties['len_l'] / self.__discreteness  # шаг вдоль кривой

        x = [xgmin]
        while True:
            X = x[-1] + step * tan2cos(derivative(self.__Fl, x[-1]))
            if X > xgmax: break
            x.append(X)
        x = array(x + [xgmax])

        Au, _, Cu = line_coefs(func=self.__Fl, x0=x)

        def equations(vars, *args):
            """СНЛАУ"""
            x0, y0, r0, xl = vars
            xu, yu, Au, Cu = args

            Al, _, Cl = line_coefs(func=Fu, x0=xl)

            return [abs(Au * x0 + (-1) * y0 + Cu) / sqrt(Au ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xu - x0) ** 2 + (yu - y0) ** 2) - r0 ** 2,  # до кривой корыта
                    abs(Al * x0 + (-1) * y0 + Cl) / sqrt(Al ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xl - x0) ** 2 + (Fu(xl) - y0) ** 2) - r0 ** 2]  # до кривой спинки

        xd, yd, d = list(), list(), list()

        warnings.filterwarnings('error')
        for xu, yu, a_u, c_u in tqdm(zip(x, self.__Fl(x), Au, Cu), desc='Channel calculation', total=len(x)):
            try:
                res = fsolve(equations, array((xu, yu, self.__relative_step / 2, xu)), args=(xu, yu, a_u, c_u))
            except Exception:
                continue

            if all((xgmin <= res[0] <= xgmax,
                    Fu(res[0]) < res[1] < self.__Fl(res[0]),  # y центра окружности лежит в канале
                    xgmin <= res[3] <= xgmax,
                    res[2] * 2 <= self.__relative_step)):
                xd.append(res[0])
                yd.append(res[1])
                d.append(res[2] * 2)
        warnings.filterwarnings('default')

        r = zeros(len(d))
        for i in range(1, len(d)): r[i] = r[i - 1] + dist((xd[i - 1], yd[i - 1]), (xd[i], yd[i]))

        self.__channel = array((xd, yd, d, r)).T

        return self.__channel

    # TODO
    def cfd(self):
        """Продувка"""
        pass

    def to_dataframe(self, bears: str = 'pandas'):
        assert bears in ('pandas', 'polars')
        if bears.strip().lower() == 'pandas':
            return pd.DataFrame(self.__coordinates, columns=('x', 'y'))
        if bears.strip().lower() == 'polars':
            return pl.DataFrame(self.__coordinates, schema=('x', 'y'), orient='row')

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


def test() -> None:
    """Тестирование"""
    print(Airfoil.__version__())

    Airfoil.help()

    Airfoil.rnd = 4
    print(f'Airfoil.rnd: {Airfoil.rnd}')

    airfoils = list()

    if 0:
        airfoils.append(Airfoil('BMSTU', 30, 1 / 1.698, radians(46.23)))

        airfoils[-1].rotation_angle = radians(110)
        airfoils[-1].relative_inlet_radius, airfoils[-1].relative_outlet_radius = 0.06, 0.03
        airfoils[-1].inlet_angle, airfoils[-1].outlet_angle = radians(20), radians(10)
        airfoils[-1].x_ray_cross = 0.4
        airfoils[-1].upper_proximity = 0.5

    if 1:
        airfoils.append(Airfoil('NACA', 40, 1 / 1.698, radians(46.23)))

        airfoils[-1].relative_thickness = 0.2
        airfoils[-1].x_relative_camber = 0.3
        airfoils[-1].relative_camber = 0.05
        airfoils[-1].closed = True

    if 1:
        airfoils.append(Airfoil('MYNK', 20, 1 / 1.698, radians(46.23)))

        airfoils[-1].mynk_coefficient = 0.1

    if 1:
        airfoils.append(Airfoil('PARSEC', 30, 1 / 1.698, radians(46.23)))

        airfoils[-1].relative_inlet_radius = 0.01
        airfoils[-1].f_b_u, airfoils[-1].f_b_l = (0.35, 0.055), (0.45, -0.006)
        airfoils[-1].d2y_dx2_u, airfoils[-1].d2y_dx2_l = -0.35, -0.2
        airfoils[-1].theta_outlet_u, airfoils[-1].theta_outlet_l = radians(-6), radians(0.05)

    if 1:
        airfoils.append(Airfoil('BEZIER', 30, 1 / 1.698, radians(46.23)))

        airfoils[-1].u = ((0.0, 0.0), (0.05, 0.100), (0.35, 0.200), (1.0, 0.0))
        airfoils[-1].l = ((0.0, 0.0), (0.05, -0.10), (0.35, -0.05), (0.5, 0.0), (1.0, 0.0))

    if 1:
        airfoils.append(Airfoil('MANUAL', 30, 1 / 1.698, radians(46.23)))

        airfoils[-1].u = ((0.0, 0.0), (0.10, 0.100), (0.35, 0.150), (0.5, 0.15), (1.0, 0.0))
        airfoils[-1].l = ((0.0, 0.0), (0.05, -0.05), (0.35, -0.05), (0.5, 0.0), (1.0, 0.0))
        airfoils[-1].deg = 3

    for airfoil in airfoils:
        airfoil.show()

        print(airfoil.to_dataframe(bears='pandas'))
        print(airfoil.to_dataframe(bears='polars'))

        print(Fore.MAGENTA + 'airfoil properties:' + Fore.RESET)
        for k, v in airfoil.properties.items(): print(f'{k}: {v}')

        print(Fore.MAGENTA + 'airfoil channel:' + Fore.RESET)
        print(f'{airfoil.channel}')

        airfoil.cfd()

        airfoil.export()


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
