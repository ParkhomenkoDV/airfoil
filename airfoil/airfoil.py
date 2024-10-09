import os
import sys
import time
from types import MappingProxyType  # неизменяемый словарь
import warnings

from tqdm import tqdm
from colorama import Fore
import pandas as pd
import numpy as np
from numpy import array, arange, linspace, zeros, full, zeros_like, full_like
from numpy import nan, isnan, inf, isinf, pi
from numpy import cos, sin, tan, arctan as atan, sqrt, floor, ceil, radians, degrees
from scipy import interpolate, integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from decorators import timeit, warns
from mathematics import derivative, Axis
from mathematics import coordinate_intersection_lines, coefficients_line, angle_between, distance, distance2line
from mathematics import cot, tan2cos, tan2sin

HERE = os.path.dirname(__file__)
sys.path.append(HERE)

from curves import bernstein_curve

# Список использованной литературы
REFERENCES = MappingProxyType({
    1: '''Теория и проектирование газовой турбины: учебное пособие /
В.Е. Михальцев, В.Д. Моляков; под ред. А.Ю. Вараксина. -
Москва: Издательство МГТУ им. Н.Э. Баумана, 2020. - 230, [2] с.: ил.''',
})

# словарь терминов их описания, единицы измерения и граничные значения
VOCABULARY = MappingProxyType({
    'rotation_angle': {
        'description': 'угол поворота потока',
        'unit': '[рад]',
        'type': (int, float, np.number),
        'assert': (lambda rotation_angle: '' if 0 < rotation_angle <= pi else f'0 < rotation_angle <= {pi}',), },
    'relative_inlet_radius': {
        'description': 'относительный радиус входной кромки',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda relative_inlet_radius:
                   '' if 0 <= relative_inlet_radius < 1 else '0 <= relative_inlet_radius < 1',), },
    'relative_outlet_radius': {
        'description': 'относительный радиус выходной кромки',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda relative_outlet_radius:
                   '' if 0 <= relative_outlet_radius < 1 else '0 <= relative_outlet_radius < 1',), },
    'closed': {
        'description': 'замкнутость профиля',
        'unit': '[]',
        'type': (bool,),
        'assert': tuple(), },
    'inlet_angle': {
        'description': 'угол раскрытия входной кромки',
        'unit': '[рад]',
        'type': (int, float, np.number),
        'assert': (lambda inlet_angle: '' if 0 <= inlet_angle < pi else f'0 <= inlet_angle < {pi}',), },
    'outlet_angle': {
        'description': 'угол раскрытия выходной кромки',
        'unit': '[рад]',
        'type': (int, float, np.number),
        'assert': (lambda outlet_angle: '' if 0 <= outlet_angle < pi else f'0 <= outlet_angle < {pi}',), },
    'x_ray_cross': {
        'description': 'относительная координата х пересечения входного и выходного лучей',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda x_ray_cross: '' if 0 < x_ray_cross < 1 else '0 < x_ray_cross < 1',), },
    'upper_proximity': {
        'description': 'степень приближенности к спинке',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda upper_proximity: '' if 0 <= upper_proximity <= 1 else '0 <= upper_proximity <= 1',), },
    'relative_thickness': {
        'description': 'максимальная относительная толщина',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda relative_thickness: '' if 0 <= relative_thickness < 1 else '0 <= relative_thickness < 1',), },
    'x_relative_camber': {
        'description': 'относительна координата х максимальной выпуклости',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda x_relative_camber: '' if 0 < x_relative_camber < 1 else '0 < x_relative_camber < 1',), },
    'relative_camber': {
        'description': 'относительная максимальная выпуклость',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda relative_camber: '' if 0 <= relative_camber < 1 else '0 <= relative_camber < 1',), },
    'mynk_coefficient': {
        'description': 'коэффициент Мунка',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda mynk_coefficient: '' if 0 <= mynk_coefficient else '0 <= mynk_coefficient',), },
    'x_relative_camber_upper': {
        'description': 'относительна координата х максимальной выпуклости спинки',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda x_relative_camber_upper:
                   '' if 0 < x_relative_camber_upper < 1 else '0 < x_relative_camber_upper < 1',), },
    'x_relative_camber_lower': {
        'description': 'относительна координата х максимальной выпуклости корыта',
        'unit': '[]',
        'type': (float, np.floating),
        'assert': (lambda x_relative_camber_lower:
                   '' if 0 < x_relative_camber_lower < 1 else '0 < x_relative_camber_lower < 1',), },
    'relative_camber_upper': {
        'description': 'максимальная относительная толщина спинки относительно оси х',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda relative_camber_upper:
                   '' if -1 < relative_camber_upper < 1 else '-1 < relative_camber_upper < 1',), },
    'relative_camber_lower': {
        'description': 'максимальная относительная толщина корыта относительно оси х',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda relative_camber_lower:
                   '' if -1 < relative_camber_lower < 1 else '-1 < relative_camber_lower < 1',), },
    'd2y_dx2_upper': {
        'description': 'кривизна спинки (вторая производная поверхности)',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': tuple(), },
    'd2y_dx2_lower': {
        'description': 'кривизна корыта (вторая производная поверхности)',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': tuple(), },
    'theta_outlet_upper': {
        'description': 'угол выхода между поверхностью спинки и горизонталью',
        'unit': '[]',
        'type': (int, float, np.number),
        'assert': (lambda theta_outlet_upper:
                   '' if -pi / 2 < theta_outlet_upper < pi / 2 else f'{-pi / 2} < theta_outlet_upper < {pi / 2}',), },
    'theta_outlet_lower': {
        'description': 'угол выхода между поверхностью корыта и горизонталью',
        'unit': '[рад]',
        'type': (int, float, np.number),
        'assert': (lambda theta_outlet_lower:
                   '' if -pi / 2 < theta_outlet_lower < pi / 2 else f'{-pi / 2} < theta_outlet_lower < {pi / 2}',), },
    'points': {
        'description': 'координаты полюсов',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda points: '' if 3 <= len(points) else '3 <= len(points)',
                   lambda points: '' if all(isinstance(coord, (tuple, list, np.ndarray)) for coord in points)
                   else 'all(isinstance(coord, (tuple, list, np.ndarray)) for coord in points)',
                   lambda points: '' if all(len(coord) == 2 for coord in points)
                   else 'all(len(coord) == 2 for coord in points)',
                   lambda points:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in points)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)) '
                        'for x, y in points)'), },
    'upper': {
        'description': 'координаты спинки',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda upper: '' if 3 <= len(upper) else '3 <= len(upper)',
                   lambda upper: '' if all(isinstance(coord, (tuple, list, np.ndarray)) for coord in upper)
                   else 'all(isinstance(coord, (tuple, list, np.ndarray)) for coord in upper)',
                   lambda upper: '' if all(len(coord) == 2 for coord in upper)
                   else 'all(len(coord) == 2 for coord in upper)',
                   lambda upper:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in upper)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))'
                        'for x, y in upper)',
                   lambda upper:
                   '' if tuple(upper) == tuple(sorted(upper, key=lambda point: point[0], reverse=False))
                   else 'tuple(upper) == tuple(sorted(upper, key=lambda point: point[0], reverse=False))',), },
    'lower': {
        'description': 'координаты корыта',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda lower: '' if 3 <= len(lower) else '3 <= len(lower)',
                   lambda lower: '' if all(isinstance(coord, (tuple, list, np.ndarray)) for coord in lower)
                   else 'all(isinstance(coord, (tuple, list, np.ndarray)) for coord in lower)',
                   lambda lower: '' if all(len(coord) == 2 for coord in lower)
                   else 'all(len(coord) == 2 for coord in lower)',
                   lambda lower:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in lower)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))'
                        'for x, y in lower)',
                   lambda lower:
                   '' if tuple(lower) == tuple(sorted(lower, key=lambda point: point[0], reverse=False))
                   else 'tuple(lower) == tuple(sorted(lower, key=lambda point: point[0], reverse=False))',), },
    'deg': {
        'description': 'степень интерполяции полинома',
        'unit': '[]',
        'type': (int, np.integer),
        'assert': (lambda deg: '' if 1 <= deg <= 3 else '1 <= deg <= 3',), },
    'relative_circles': {
        'description': 'относительные окружности профиля или канала',
        'unit': '[]',
        'type': (tuple, list, np.ndarray),
        'assert': (lambda relative_circles: '' if 3 <= len(relative_circles) else '3 <= len(relative_circles)',
                   lambda relative_circles:
                   '' if all(isinstance(coord, (tuple, list, np.ndarray)) for coord in relative_circles)
                   else 'all(isinstance(coord, (tuple, list, np.ndarray)) for coord in relative_circles)',
                   lambda relative_circles: '' if all(len(coord) == 2 for coord in relative_circles)
                   else 'all(len(coord) == 2 for coord in relative_circles)',
                   lambda relative_circles:
                   '' if all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))
                             for x, y in relative_circles)
                   else 'all(isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number)) '
                        'for x, y in relative_circles)',
                   lambda relative_circles: '' if all(0 < x < 1 and 0 < y for x, y in relative_circles)
                   else 'all(0 < x < 1 and 0 < y for x, y in relative_circles)',), },
    'is_airfoil': {
        'description': 'указатель на профиль',
        'unit': '[]',
        'type': (bool,),
        'assert': tuple(), },
})


class Airfoil:
    """Относительный аэродинамический профиль"""

    __rnd = 4  # количество значащих цифр
    __discreteness = 30  # рекомендуемое количество дискретных точек
    # TODO
    __methods = {
        'BMSTU': {
            'description': '',
            'aliases': ('BMSTU', 'МГТУ', 'МВТУ', 'МИХАЛЬЦЕВ'),
            'attributes': {
                'rotation_angle': VOCABULARY['rotation_angle'],
                'relative_inlet_radius': VOCABULARY['relative_inlet_radius'],
                'relative_outlet_radius': VOCABULARY['relative_outlet_radius'],
                'inlet_angle': VOCABULARY['inlet_angle'],
                'outlet_angle': VOCABULARY['outlet_angle'],
                'x_ray_cross': VOCABULARY['x_ray_cross'],
                'upper_proximity': VOCABULARY['upper_proximity']}},
        'NACA': {'description': '',
                 'aliases': ('NACA', 'N.A.C.A.'),
                 'attributes': {
                     'relative_thickness': VOCABULARY['relative_thickness'],
                     'x_relative_camber': VOCABULARY['x_relative_camber'],
                     'relative_camber': VOCABULARY['relative_camber'],
                     'closed': VOCABULARY['closed'], }},
        'MYNK': {'description': '',
                 'aliases': ('MYNK', 'МУНК'),
                 'attributes': {
                     'mynk_coefficient': VOCABULARY['mynk_coefficient'], }},
        'PARSEC': {'description': '',
                   'aliases': ('PARSEC',),
                   'attributes': {
                       'relative_inlet_radius': VOCABULARY['relative_inlet_radius'],
                       'x_relative_camber_upper': VOCABULARY['x_relative_camber_upper'],
                       'x_relative_camber_lower': VOCABULARY['x_relative_camber_lower'],
                       'relative_camber_upper': VOCABULARY['relative_camber_upper'],
                       'relative_camber_lower': VOCABULARY['relative_camber_lower'],
                       'd2y_dx2_upper': VOCABULARY['d2y_dx2_upper'],
                       'd2y_dx2_lower': VOCABULARY['d2y_dx2_lower'],
                       'theta_outlet_upper': VOCABULARY['theta_outlet_upper'],
                       'theta_outlet_lower': VOCABULARY['theta_outlet_lower'], }},
        'BEZIER': {'description': '',
                   'aliases': ('BEZIER', 'БЕЗЬЕ'),
                   'attributes': {
                       'points': VOCABULARY['points'], }},
        'MANUAL': {'description': '',
                   'aliases': ('MANUAL', 'SPLINE', 'ВРУЧНУЮ'),
                   'attributes': {
                       'upper': VOCABULARY['upper'],
                       'lower': VOCABULARY['lower'],
                       'deg': VOCABULARY['deg'], }},
        'CIRCLE': {'description': '',
                   'aliases': ('CIRCLE', 'ОКРУЖНОСТЬ',),
                   'attributes': {
                       'relative_circles': VOCABULARY['relative_circles'],
                       'rotation_angle': VOCABULARY['rotation_angle'],
                       'x_ray_cross': VOCABULARY['x_ray_cross'],
                       'is_airfoil': VOCABULARY['is_airfoil'], }}, }
    __relative_step = 1.0  # дефолтный относительный шаг []
    __installation_angle = 0.0  # дефолтный угол установки [рад]

    @classmethod
    @property
    def __version__(cls) -> str:
        version = '4.0'
        todo = ('Дописать help', 'Сделать продувку')
        for i, td in enumerate(todo): print(float(version) + (i + 1), td)
        return version

    @property
    def rnd(self) -> int:
        return Airfoil.__rnd

    @rnd.setter
    def rnd(self, value) -> None:
        assert isinstance(value, int) and 0 <= value
        Airfoil.__rnd = value

    @rnd.deleter
    def rnd(self):
        raise

    # TODO
    @classmethod
    def help(cls):
        """Помощь при работе с классом Airfoil и его объектами"""
        print(Fore.MAGENTA + 'Airfoil tutorial' + Fore.RESET)
        print('Airfoil.rnd = 4  # количество значащих цифр')
        print('Airfoil.vocabulary  # словарь терминов и атрибутов')
        print()
        print('airfoil = Airfoil(method, discreteness, relative_step, gamma, **attributes)  # создание объекта')
        print('где:')
        print('discreteness:  int >= 3      # количество дискретных точек')
        print('relative_step: int > 0       # относительный шаг')
        print('gamma:         float < pi/2  # угол установки профиля')
        print()
        print('methods:  # методы построения аэродинамического профиля')
        for method in Airfoil.__methods:
            print(method)
            print('\t' + f'description: {Airfoil.__methods[method]["description"]}')
            print('\t' + f'aliases: {Airfoil.__methods[method]["aliases"]}')
            print('\t' + f'attributes:')
            for attribute in Airfoil.__methods[method]["attributes"]:
                print('\t\t' + Fore.CYAN + f'{attribute}' + Fore.RESET)
                for key, value in Airfoil.__methods[method]["attributes"][attribute].items():
                    print('\t\t\t' + f'{key}: {value}')
        print()
        print('airfoil.show()')
        print('airfoil.properties')
        print('airfoil.channel')
        print('airfoil.to_dataframe()')
        print('airfoil.export()')

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

        if hasattr(self, '_Airfoil__method'):
            for attr in Airfoil.__methods[self.__method]['attributes']:
                assert hasattr(self, attr), f'not hasattr({attr})'
                assert isinstance(getattr(self, attr), Airfoil.__methods[self.__method]['attributes'][attr]['type']), \
                    f'type({attr}) not in {Airfoil.__methods[self.__method]["attributes"][attr]["type"]}'
                for ass in Airfoil.__methods[self.__method]['attributes'][attr]["assert"]:
                    assert not ass(getattr(self, attr)), ass(getattr(self, attr))

    def __init__(self, method: str, discreteness: int = __discreteness,
                 relative_step: float | int = __relative_step, installation_angle: float | int = __installation_angle,
                 **attributes):
        self.validate(method=method, discreteness=discreteness,
                      relative_step=relative_step, installation_angle=installation_angle)

        self.__method = method.strip().upper()  # метод построения аэродинамического профиля
        self.__discreteness = discreteness  # количество точек дискретизации

        self.__relative_step = relative_step  # относительный шаг []
        self.__installation_angle = installation_angle  # угол установки [рад]

        self.__coordinates = tuple()  # относительные координаты профиля считая против часовой стрелки с выходной кромки
        self.__properties = dict()  # относительные характеристики профиля
        self.__channel = tuple()  # дифузорность/конфузорность решетки

        for attribute, value in attributes.items(): setattr(self, attribute, value)

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
    def installation_angle(self) -> float | int | np.number:
        return self.__installation_angle

    @installation_angle.setter
    def installation_angle(self, installation_angle):
        self.validate(installation_angle=installation_angle)
        self.__init__(method=self.method, installation_angle=installation_angle)

    @installation_angle.deleter
    def installation_angle(self):
        self.__installation_angle = Airfoil.__installation_angle

    @property
    def coordinates(self) -> tuple[tuple[float, float], ...]:
        if len(self.__coordinates) == 0: self.__calculate()
        return self.__coordinates

    # TODO
    def input(self):
        """Динамический ввод с защитой от дураков"""
        pass

    def __bmstu(self) -> tuple[tuple[float, float], ...]:
        airfoil_rotation_angle = pi - self.rotation_angle  # угол поворота профиля

        # tan угла входа и выхода потока
        k_inlet = 1 / (2 * self.x_ray_cross / (self.x_ray_cross - 1) * tan(airfoil_rotation_angle))
        k_outlet = 1 / (2 * tan(airfoil_rotation_angle))
        if tan(airfoil_rotation_angle) * airfoil_rotation_angle > 0:
            k_inlet *= ((self.x_ray_cross / (self.x_ray_cross - 1) - 1) -
                        sqrt((self.x_ray_cross / (self.x_ray_cross - 1) - 1) ** 2 -
                             4 * (self.x_ray_cross / (self.x_ray_cross - 1) * tan(airfoil_rotation_angle) ** 2)))
            k_outlet *= ((self.x_ray_cross / (self.x_ray_cross - 1) - 1) -
                         sqrt((self.x_ray_cross / (self.x_ray_cross - 1) - 1) ** 2 -
                              4 * (self.x_ray_cross / (self.x_ray_cross - 1) * tan(airfoil_rotation_angle) ** 2)))
        else:
            k_inlet *= ((self.x_ray_cross / (self.x_ray_cross - 1) - 1) +
                        sqrt((self.x_ray_cross / (self.x_ray_cross - 1) - 1) ** 2 -
                             4 * (self.x_ray_cross / (self.x_ray_cross - 1) * tan(airfoil_rotation_angle) ** 2)))
            k_outlet *= ((self.x_ray_cross / (self.x_ray_cross - 1) - 1) +
                         sqrt((self.x_ray_cross / (self.x_ray_cross - 1) - 1) ** 2 -
                              4 * (self.x_ray_cross / (self.x_ray_cross - 1) * tan(airfoil_rotation_angle) ** 2)))

        # углы входа и выхода профиля
        if airfoil_rotation_angle > 0:
            g_u_inlet, g_d_inlet = ((1 - self.upper_proximity) *
                                    self.inlet_angle, self.upper_proximity * self.inlet_angle)
            g_u_outlet, g_d_outlet = ((1 - self.upper_proximity) *
                                      self.outlet_angle, self.upper_proximity * self.outlet_angle)
        else:
            g_u_inlet, g_d_inlet = self.upper_proximity * self.inlet_angle, (
                    1 - self.upper_proximity) * self.inlet_angle,
            g_u_outlet, g_d_outlet = self.upper_proximity * self.outlet_angle, (
                    1 - self.upper_proximity) * self.outlet_angle

        # положения центров окружностей входной и выходной кромок
        O_inlet = self.relative_inlet_radius, k_inlet * self.relative_inlet_radius
        O_outlet = 1 - self.relative_outlet_radius, -k_outlet * self.relative_outlet_radius

        # точки пересечения линий спинки и корыта
        xcl_u, ycl_u = coordinate_intersection_lines(
            (tan(atan(k_inlet) + g_u_inlet), -1,
             sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * self.relative_inlet_radius -
             (tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1]),
            (tan(atan(k_outlet) - g_u_outlet), -1,
             sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * self.relative_outlet_radius -
             (tan(atan(k_outlet) - g_u_outlet)) * O_outlet[0] - (-1) * O_outlet[1]))

        xcl_d, ycl_d = coordinate_intersection_lines(
            (tan(atan(k_inlet) - g_d_inlet), -1,
             -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * self.relative_inlet_radius -
             (tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1]),
            (tan(atan(k_outlet) + g_d_outlet), -1,
             -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * self.relative_outlet_radius -
             (tan(atan(k_outlet) + g_d_outlet)) * O_outlet[0] - (-1) * O_outlet[1]))

        # точки пересечения окружностей со спинкой и корытом
        xclc_i_u, yclc_i_u = coordinate_intersection_lines(
            (tan(atan(k_inlet) + g_u_inlet), -1,
             sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * self.relative_inlet_radius
             - (tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1]),
            (-1 / (tan(atan(k_inlet) + g_u_inlet)), -1,
             -(-1 / tan(atan(k_inlet) + g_u_inlet)) * O_inlet[0] - (-1) * O_inlet[1]))

        xclc_i_d, yclc_i_d = coordinate_intersection_lines(
            (tan(atan(k_inlet) - g_d_inlet), -1,
             -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * self.relative_inlet_radius
             - (tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1]),
            (-1 / (tan(atan(k_inlet) - g_d_inlet)), -1,
             -(-1 / tan(atan(k_inlet) - g_d_inlet)) * O_inlet[0] - (-1) * O_inlet[1]))

        xclc_e_u, yclc_e_u = coordinate_intersection_lines(
            (tan(atan(k_outlet) - g_u_outlet), -1,
             sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * self.relative_outlet_radius
             - tan(atan(k_outlet) - g_u_outlet) * O_outlet[0] - (-1) * O_outlet[1]),
            (-1 / tan(atan(k_outlet) - g_u_outlet), -1,
             -(-1 / tan(atan(k_outlet) - g_u_outlet)) * O_outlet[0] - (-1) * O_outlet[1]))

        xclc_e_d, yclc_e_d = coordinate_intersection_lines(
            (tan(atan(k_outlet) + g_d_outlet), -1,
             -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * self.relative_outlet_radius
             - tan(atan(k_outlet) + g_d_outlet) * O_outlet[0] - (-1) * O_outlet[1]),
            (-1 / tan(atan(k_outlet) + g_d_outlet), -1,
             -(-1 / tan(atan(k_outlet) + g_d_outlet)) * O_outlet[0] - (-1) * O_outlet[1]))

        x, y = list(), list()

        # окружность выходной кромки спинки
        an = angle_between(points=((1, O_outlet[1]), O_outlet, (xclc_e_u, yclc_e_u)))
        if not isnan(an):  # при не нулевом радиусе окружности
            if O_outlet[0] > xclc_e_u: an = pi - an
            # уменьшение угла для предотвращения дублирования координат
            angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
            x += (1 - self.relative_outlet_radius * (1 - cos(angles))).tolist()
            y += (O_outlet[1] + self.relative_outlet_radius * sin(angles)).tolist()

        # спинка
        xu, yu = bernstein_curve(((xclc_e_u, yclc_e_u), (xcl_u, ycl_u), (xclc_i_u, yclc_i_u)),
                                 N=self.__discreteness).T.tolist()
        x += xu
        y += yu

        # точки входной окружности кромки по спинке
        an = angle_between(points=((0, O_inlet[1]), O_inlet, (xclc_i_u, yclc_i_u)))
        if not isnan(an):  # при не нулевом радиусе окружности
            if xclc_i_u > O_inlet[0]: an = pi - an
            # уменьшение угла для предотвращения дублирования координат
            angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
            x += (self.relative_inlet_radius * (1 - cos(angles))).tolist()[::-1]
            y += (O_inlet[1] + self.relative_inlet_radius * sin(angles)).tolist()[::-1]

        x.pop(), y.pop()  # удаление дубликата входной точки

        # окружность входной кромки корыта
        an = angle_between(points=((0, O_inlet[1]), O_inlet, (xclc_i_d, yclc_i_d)))
        if not isnan(an):  # при не нулевом радиусе окружности
            if xclc_i_d > O_inlet[0]: an = pi - an
            # уменьшение угла для предотвращения дублирования координат
            angles = linspace(0, an * 0.99, self.discreteness, endpoint=False)
            x += (self.relative_inlet_radius * (1 - cos(angles))).tolist()
            y += (O_inlet[1] - self.relative_inlet_radius * sin(angles)).tolist()

        # корыто
        xd, yd = bernstein_curve(((xclc_i_d, yclc_i_d), (xcl_d, ycl_d), (xclc_e_d, yclc_e_d)),
                                 N=self.__discreteness).T.tolist()
        x += xd
        y += yd

        # точки выходной окружности кромки по корыту
        an = angle_between(points=((1, O_outlet[1]), O_outlet, (xclc_e_d, yclc_e_d)))
        if not isnan(an):  # при не нулевом радиусе окружности
            if O_outlet[0] > xclc_e_d: an = pi - an
            # уменьшение угла для предотвращения дублирования координат
            angles = linspace(0, an * 0.99, self.__discreteness, endpoint=False)
            x += (1 - self.relative_outlet_radius * (1 - cos(angles))).tolist()[::-1]
            y += (O_outlet[1] - self.relative_outlet_radius * sin(angles)).tolist()[::-1]

        return tuple(((x, y) for x, y in zip(x, y)))

    def __naca(self) -> tuple[tuple[float, float], ...]:
        i = arange(self.__discreteness)  # массив индексов
        betta = i * pi / (2 * (self.__discreteness - 1))
        x = 1 - cos(betta)

        mask = (0 <= x) & (x <= self.x_relative_camber)

        yf = full_like(i, self.relative_camber, dtype='float64')
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

        scale = abs(X.max() - X.min())

        return self.transform(tuple(((x, y) for x, y in zip(X, Y))), x0=X.min(), scale=(1 / scale))

    def __mynk(self) -> tuple[tuple[float, float], ...]:

        def mynk_coordinates(param: float, x) -> tuple:
            """Координата y спинки и корыта"""
            part1, part2 = 0.25 * (-x - 17 * x ** 2 - 6 * x ** 3), x ** 0.87 * (1 - x) ** 0.56
            return param * (part1 + part2), param * (part1 - part2)

        x = linspace(0, 1, self.__discreteness, endpoint=True)
        yu, yl = mynk_coordinates(self.mynk_coefficient, x)
        idx = np.argmax(yu)
        angle = atan((yu[-1] - yu[0]) / (x[-1] - x[0]))

        X, Y = np.hstack((x[-1:idx:-1], x[idx::-1], x[1::])), np.hstack((yu[-1:idx:-1], yu[idx::-1], yl[1::]))

        coordinates = self.transform(tuple(((x, y) for x, y in zip(X, Y))), angle=angle)  # поворот
        x, _ = array(coordinates).T
        return self.transform(coordinates, x0=x.min(), scale=(1 / (x.max() - x.min())))  # нормализация

    def __parsec(self) -> tuple[tuple[float, float], ...]:
        """
        Generate and plot the contour of an airfoil using the PARSEC parameterization
        H. Sobieczky, *'Parametric airfoils and wings'* in *Notes on Numerical Fluid Mechanics*, Vol. 68, pp 71-88]
        (www.as.dlr.de/hs/h-pdf/H141.pdf)
        Repository & documentation: http://github.com/dqsis/parsec-airfoils
        """

        def parsec_coefficients(surface: str,
                                radius_inlet: float | int,
                                c_b: tuple[float, float], d2y_dx2_surface,
                                outlet: tuple, theta_outlet_surface: float | int):
            """PARSEC coefficients"""
            assert surface in ('l', 'u')
            assert isinstance(c_b, (tuple, list)) and len(c_b) == 2
            assert isinstance(outlet, (tuple, list)) and len(outlet) == 2

            x_c_b, y_c_b = c_b
            x_outlet, y_outlet = outlet

            coefs = zeros(6)

            # 1-ый коэффициент зависит от кривой поверхности спинки или корыта
            coefs[0] = -sqrt(2 * radius_inlet) if surface == 'l' else sqrt(2 * radius_inlet)

            i = arange(1, 6)

            # матрицы коэффициентов системы уравнений
            A = array([x_outlet ** (i + 0.5),
                       x_c_b ** (i + 0.5),
                       (i + 0.5) * x_outlet ** (i - 0.5),
                       (i + 0.5) * x_c_b ** (i - 0.5),
                       (i ** 2 - 0.25) * x_c_b ** (i - 1.5)])
            B = array([[y_outlet - coefs[0] * sqrt(x_outlet)],
                       [y_c_b - coefs[0] * sqrt(x_c_b)],
                       [tan(theta_outlet_surface) - 0.5 * coefs[0] * (1 / sqrt(x_outlet))],
                       [-0.5 * coefs[0] * (1 / sqrt(x_c_b))],
                       [d2y_dx2_surface + 0.25 * coefs[0] * x_c_b ** (-1.5)]])

            X = np.linalg.solve(A, B)  # решение СЛАУ
            coefs[1:6] = X[0:5, 0]  # 0 коэффициент уже есть

            return coefs

        # поверхностные коэффициенты давления спинки и корыта
        coefs_u = parsec_coefficients('u', self.relative_inlet_radius,
                                      (self.x_relative_camber_upper, self.relative_camber_upper), self.d2y_dx2_upper,
                                      (1, 0), self.theta_outlet_upper)
        coefs_l = parsec_coefficients('l', self.relative_inlet_radius,
                                      (self.x_relative_camber_lower, self.relative_camber_lower), self.d2y_dx2_lower,
                                      (1, 0), self.theta_outlet_lower)

        x = linspace(0, 1, self.__discreteness, endpoint=True)

        X, Y = np.hstack((x[::-1], x[1::])), np.hstack((sum([coefs_u[i] * x ** (i + 0.5) for i in range(6)])[::-1],
                                                        sum([coefs_l[i] * x ** (i + 0.5) for i in range(6)])[1::]))
        return tuple((x, y) for x, y in zip(X, Y))

    def __bezier(self) -> tuple[tuple[float, float], ...]:
        X, Y = bernstein_curve(self.points, N=self.__discreteness).T
        xargmin, xargmax = np.argmin(X), np.argmax(X)
        angle = atan((Y[xargmax] - Y[xargmin]) / (X[xargmax] - X[xargmin]))  # угол поворота
        coordinates = self.transform(tuple(((x, y) for x, y in zip(X, Y))), angle=angle)  # поворот
        x, y = array(coordinates).T
        return self.transform(coordinates, x0=x.min(), y0=y[0], scale=(1 / (x.max() - x.min())))  # нормализация

    def __manual(self) -> tuple[tuple[float, float], ...]:
        xu, yu = array(self.upper, dtype='float64').T
        xl, yl = array(self.lower, dtype='float64').T

        xmin, xmax = min(xu.min(), xl.min()), max(xu.max(), xl.max())
        scale = xmax - xmin

        xl, xu = (xl - xmin) / scale, (xu - xmin) / scale

        fu, fl = interpolate.interp1d(xu, yu, kind=self.deg), interpolate.interp1d(xl, yl, kind=self.deg)
        X = linspace(0, 1, self.__discreteness, endpoint=True)
        coordinates = [(x, y) for x, y in zip(X[::-1], fu(X[::-1]))] + [(x, y) for x, y in zip(X[1::], fl(X[1::]))]
        return tuple(coordinates)

    def __circle(self) -> tuple[tuple[float, float], ...]:

        y_ray_cross = fsolve(lambda y:
                             atan(self.x_ray_cross / y) + atan((1 - self.x_ray_cross) / y) - pi + self.rotation_angle,
                             array(0.5, dtype='float64'))[0]

        # точки безье средней линии, на которые будут накладываться окружности
        x_av, y_av = bernstein_curve(((0, 0), (self.x_ray_cross, y_ray_cross), (1, 0)), N=self.__discreteness).T
        f_av = interpolate.interp1d(x_av, y_av, kind=3, fill_value='extrapolate')  # функция средней линии

        # длина кривой центров окружностей
        l = integrate.quad(lambda x: sqrt(1 + derivative(f_av, x) ** 2), 0, 1, epsrel=0.000_1)[0]
        step = l / self.__discreteness  # шаг по кривой центров окружностей

        xc, dc = array(self.relative_circles, dtype='float64').T

        if self.is_airfoil:
            xc = np.insert(xc, 0, 0)  # профиль в отличие от канала должен быть замкнутым с концов
            dc = np.insert(dc, 0, 0)  # а для канала в точках х 0 и 1 неизвестны окружности
            xc = np.append(xc, 1)
            dc = np.append(dc, 0)

        xc *= l  # масштабирование хc по длине l

        # непрерывная функция диаметров окружностей по длине средней линии
        f_d = interpolate.interp1d(xc, dc, kind=1, fill_value='extrapolate')  # kind=1 для безопасности

        dy_dx, x_circle, X = list(), list(), -0.5
        while X <= 1.5:
            x_circle.append(X)
            dy_dx.append(derivative(f_av, X))
            X = x_circle[-1] + step * tan2cos(dy_dx[-1])
        x_circle = array(x_circle, dtype='float64')
        y_circle = f_av(x_circle)
        d_circle = np.maximum(f_d(x_circle), 0)  # интерполяция ушла в минус
        dy_dx_ = -1 / array(dy_dx, dtype='float64')  # перпендикуляры

        mask = dy_dx_ >= 0  # маска положительного наклона перпендикуляра
        xu, yu, xl, yl = x_circle.copy(), y_circle.copy(), x_circle.copy(), y_circle.copy()
        d_2_tan2cos, d_2_tan2sin = d_circle / 2 * tan2cos(dy_dx_), d_circle / 2 * tan2sin(dy_dx_)

        xu[mask] += d_2_tan2cos[mask]
        yu[mask] += d_2_tan2sin[mask]
        xl[mask] -= d_2_tan2cos[mask]
        yl[mask] -= d_2_tan2sin[mask]

        xu[~mask] -= d_2_tan2cos[~mask]
        yu[~mask] -= d_2_tan2sin[~mask]
        xl[~mask] += d_2_tan2cos[~mask]
        yl[~mask] += d_2_tan2sin[~mask]

        upper_mask, lower_mask = (0 < xu) & (xu < 1), (0 < xl) & (xl < 1)  # маска принадлежности к интервалу (0, 1)
        xu, yu, xl, yl = xu[upper_mask], yu[upper_mask], xl[lower_mask], yl[lower_mask]

        if not self.is_airfoil:
            yu -= self.relative_step  # перенос спинки вниз
            yu, yl, xu, xl = yl, yu, xl, xu  # правильное обозначение

        X = np.hstack([[1], xu[::-1], [0], xl, [1]])
        Y = np.hstack([[0], yu[::-1], [0], yl, [0]]) if self.is_airfoil \
            else np.hstack([[- self.relative_step / 2],
                            yu[::-1],
                            [- self.relative_step / 2],
                            yl,
                            [- self.relative_step / 2]])

        return tuple((x, y) for x, y in zip(X, Y))

    def __calculate(self) -> tuple[tuple[float, float], ...]:
        self.validate()

        if self.method in Airfoil.__methods['NACA']['aliases']:
            self.__coordinates0 = self.__naca()
            self.__relative_outlet_radius = 0 if self.closed \
                else abs(self.__coordinates0[0][1] - self.__coordinates0[-1][1])
        elif self.method in Airfoil.__methods['BMSTU']['aliases']:
            self.__coordinates0 = self.__bmstu()
            self.__relative_inlet_radius = self.relative_inlet_radius
            self.__relative_outlet_radius = self.relative_outlet_radius
        elif self.method in Airfoil.__methods['MYNK']['aliases']:
            self.__coordinates0 = self.__mynk()
            self.__relative_inlet_radius, self.__relative_outlet_radius = 0, 0
        elif self.method in Airfoil.__methods['PARSEC']['aliases']:
            self.__coordinates0 = self.__parsec()
            self.__relative_inlet_radius, self.__relative_outlet_radius = 0, 0
        elif self.method in Airfoil.__methods['BEZIER']['aliases']:
            self.__coordinates0 = self.__bezier()
            self.__relative_inlet_radius, self.__relative_outlet_radius = 0, 0
        elif self.method in Airfoil.__methods['MANUAL']['aliases']:
            self.__coordinates0 = self.__manual()
            self.__relative_inlet_radius, self.__relative_outlet_radius = 0, 0
        elif self.method in Airfoil.__methods['CIRCLE']['aliases']:
            self.__coordinates0 = self.__circle()
            self.__relative_inlet_radius, self.__relative_outlet_radius = 0, 0
        else:
            print(Fore.RED + f'No such method {self.method}! Use Airfoil.help' + Fore.RESET)

        self.__coordinates = self.transform(self.__coordinates0, angle=self.__installation_angle)  # поворот
        coordinates = array(self.__coordinates, dtype='float64').T
        x_min, x_max = coordinates[0].min(), coordinates[0].max()
        scale = abs(x_max - x_min)
        self.__coordinates = self.transform(self.__coordinates, x0=x_min, scale=(1 / scale))  # нормал
        return self.__coordinates

    def transform(self, coordinates: tuple[tuple[float, float], ...],
                  x0=0.0, y0=0.0, angle=0.0, scale=1.0) -> tuple[tuple[float, float], ...]:
        """Перенос-поворот кривых спинки и корыта профиля"""

        new_coordinates = list()
        for x, y in coordinates:
            point = Axis.transform(x, y, x0=x0, y0=y0, angle=angle, scale=scale)
            new_coordinates.append((float(point[0]), float(point[1])))

        return tuple(new_coordinates)

    @staticmethod
    def upper_lower(coordinates: tuple[tuple[float, float], ...]) -> dict[str:tuple[tuple[float, float], ...]]:
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

        coordinates = self.upper_lower(coordinates)

        fu = interpolate.interp1d(*array(coordinates['upper']).T, kind=3, fill_value='extrapolate')
        fl = interpolate.interp1d(*array(coordinates['lower']).T, kind=3, fill_value='extrapolate')

        x0, x1 = 0, 1  # координаты x входной и выходной окружности
        y0, y1 = fu(x0), fu(x1)  # координаты y входной и выходной окружности

        x0u = x0 + dl * tan2cos(derivative(fu, x0, method='forward', dx=dl))
        x1u = x1 - dl * tan2cos(derivative(fu, x1, method='backward', dx=dl))
        y0u, y1u = fu(x0u), fu(x1u)
        x0l = x0 + dl * tan2cos(derivative(fl, x0, method='forward', dx=dl))
        x1l = x1 - dl * tan2cos(derivative(fl, x1, method='backward', dx=dl))
        y0l, y1l = fl(x0l), fl(x1l)

        A0u, B0u, C0u = coefficients_line(p1=(x0, y0), p2=(x0u, y0u))
        A0l, B0l, C0l = coefficients_line(p1=(x0, y0), p2=(x0l, y0l))
        A1u, B1u, C1u = coefficients_line(p1=(x1, y1), p2=(x1u, y1u))
        A1l, B1l, C1l = coefficients_line(p1=(x1, y1), p2=(x1l, y1l))

        # коэффициент A для перпендикуляров
        AA0u, AA0l = -1 / A0u if A0u != 0 else -inf, -1 / A0l if A0l != 0 else -inf
        AA1u, AA1l = -1 / A1u if A1u != 0 else -inf, -1 / A1l if A1l != 0 else -inf
        # коэффициент С для перпендикуляров
        CC0u, CC0l = np.mean((y0, y0u)) - AA0u * np.mean((x0, x0u)), np.mean((y0, y0l)) - AA0l * np.mean((x0, x0l))
        CC1u, CC1l = np.mean((y1, y1u)) - AA1u * np.mean((x1, x1u)), np.mean((y1, y1l)) - AA1l * np.mean((x1, x1l))

        # центры входной и выходной окружностей
        O_inlet = coordinate_intersection_lines((AA0u, -1, CC0u), (AA0l, -1, CC0l))
        O_outlet = coordinate_intersection_lines((AA1u, -1, CC1u), (AA1l, -1, CC1l))
        if not (0.0 <= O_inlet[0] <= 0.5) or not (fl(O_inlet[0]) <= O_inlet[1] <= fu(O_inlet[0])):
            O_inlet = (nan, nan)
        if not (0.5 <= O_outlet[0] <= 1.0) or not (fl(O_outlet[0]) <= O_outlet[1] <= fu(O_outlet[0])):
            O_outlet = (nan, nan)

        if not hasattr(self, '_Airfoil__relative_inlet_radius'): self.__relative_inlet_radius = abs(O_inlet[0] - x0)
        if not hasattr(self, '_Airfoil__relative_outlet_radius'): self.__relative_outlet_radius = abs(O_outlet[0] - x1)

        return {'inlet': {'point': O_inlet, 'radius': self.__relative_inlet_radius},
                'outlet': {'point': O_outlet, 'radius': self.__relative_outlet_radius}}

    def show(self, amount: int = 2, figsize=(12, 10), savefig=False):
        """Построение профиля"""
        assert isinstance(amount, int) and 1 <= amount  # количество профилей

        X, Y = array(self.coordinates).T  # запуск расчета
        coordinates0 = self.upper_lower(self.__coordinates0)
        x, y, d, r = self.channel.T

        fg = plt.figure(figsize=figsize)
        gs = fg.add_gridspec(nrows=2, ncols=3)

        fg.add_subplot(gs[0, 0])
        plt.title('Initial data')
        plt.axis('off')
        plt.plot([], label=f'method = {self.method}')
        plt.plot([], label=f'discreteness = {self.__discreteness}')
        plt.plot([], label=f'relative_step = {self.__relative_step:.{Airfoil.__rnd}f} []')
        plt.plot([],
                 label=f'gamma = {self.__installation_angle:.{Airfoil.__rnd}f} [rad] = {degrees(self.__installation_angle):.{Airfoil.__rnd}f} [deg]')
        for key, value in self.__dict__.items():
            if not key.startswith('_') and isinstance(value, (int, float, np.number)):
                plt.plot([], label=f'{key} = {value:.{Airfoil.__rnd}f}')
        plt.legend(loc='upper center')

        fg.add_subplot(gs[1, 0])
        plt.title('Properties')
        plt.axis('off')
        for key, value in self.properties.items(): plt.plot([], label=f'{key} = {value:.{Airfoil.__rnd}f}')
        plt.legend(loc='upper center')

        fg.add_subplot(gs[0, 1])
        plt.title('Airfoil structure')
        plt.grid(True)  # сетка
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.plot(*array(coordinates0['upper']).T, ls='solid', color='blue', linewidth=2)
        plt.plot(*array(coordinates0['lower']).T, ls='solid', color='red', linewidth=2)
        alpha = linspace(0, 2 * pi, 360, endpoint=True)
        circles = self.__find_circles(self.__coordinates0)
        x_inlet = self.__relative_inlet_radius * cos(alpha) + circles['inlet']['point'][0]
        y_inlet = self.__relative_inlet_radius * sin(alpha) + circles['inlet']['point'][1]
        x_outlet = self.__relative_outlet_radius * cos(alpha) + circles['outlet']['point'][0]
        y_outlet = self.__relative_outlet_radius * sin(alpha) + circles['outlet']['point'][1]
        plt.plot(x_inlet, y_inlet, ls='solid', color='black', linewidth=1)
        plt.plot(x_outlet, y_outlet, ls='solid', color='black', linewidth=1)

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

        plt.tight_layout()
        if savefig: plt.savefig(f'pictures/airfoil_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.png')
        plt.show()

    @property
    @timeit()
    def properties(self, epsrel: float = 1e-4) -> dict[str: float]:
        if self.__properties: return self.__properties

        if not hasattr(self, '_Airfoil__relative_inlet_radius'):
            self.__relative_inlet_radius = 0  # self.__find_circles(self.coordinates)['inlet']['radius']
        if not hasattr(self, '_Airfoil__relative_outlet_radius'):
            self.__relative_outlet_radius = 0  # self.__find_circles(self.coordinates)['outlet']['radius']
        self.__properties['radius_inlet'] = self.__relative_inlet_radius
        self.__properties['radius_outlet'] = self.__relative_outlet_radius

        dct = self.upper_lower(self.coordinates)
        self.__fu = interpolate.interp1d(*array(dct['upper']).T, kind=3, fill_value='extrapolate')
        self.__fl = interpolate.interp1d(*array(dct['lower']).T, kind=3, fill_value='extrapolate')

        self.__properties['area'] = integrate.dblquad(lambda _, __: 1,
                                                      0, 1, lambda xu: self.__fl(xu), lambda xl: self.__fu(xl),
                                                      epsrel=epsrel)[0]
        x = linspace(0, 1, int(ceil(1 / epsrel)))
        fu, fl = self.__fu(x), self.__fl(x)
        delta_f = fu - fl
        delta_f_2 = delta_f / 2
        argmax_c, argmax_f = np.argmax(delta_f), np.argmax(np.abs(delta_f_2))
        self.__properties['xc'], self.__properties['c'] = x[argmax_c], delta_f[argmax_c]
        self.__properties['xf'], self.__properties['f'] = x[argmax_f], delta_f_2[argmax_f]  # TODO неверно!
        self.__properties['Sx'] = integrate.dblquad(lambda y, _: y,
                                                    0, 1, lambda xu: self.__fl(xu), lambda xd: self.__fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Sy'] = integrate.dblquad(lambda _, x: x,
                                                    0, 1, lambda xu: self.__fl(xu), lambda xd: self.__fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['x0'] = self.__properties['Sy'] / self.__properties['area'] \
            if self.__properties['area'] != 0 else inf
        self.__properties['y0'] = self.__properties['Sx'] / self.__properties['area'] \
            if self.__properties['area'] != 0 else inf
        self.__properties['Jx'] = integrate.dblquad(lambda y, _: y ** 2,
                                                    0, 1, lambda xu: self.__fl(xu), lambda xd: self.__fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Jy'] = integrate.dblquad(lambda _, x: x ** 2,
                                                    0, 1, lambda xu: self.__fl(xu), lambda xd: self.__fu(xd),
                                                    epsrel=epsrel)[0]
        self.__properties['Jxy'] = integrate.dblquad(lambda y, x: x * y,
                                                     0, 1, lambda xu: self.__fl(xu), lambda xd: self.__fu(xd),
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
        self.__properties['len_u'] = integrate.quad(lambda x: sqrt(1 + derivative(self.__fu, x) ** 2),
                                                    0, 1,
                                                    epsrel=epsrel)[0]
        self.__properties['len_l'] = integrate.quad(lambda x: sqrt(1 + derivative(self.__fl, x) ** 2),
                                                    0, 1,
                                                    epsrel=epsrel)[0]
        return self.__properties

    @property
    @timeit()
    def channel(self) -> np.ndarray:
        """Диффузорность/конфузорность решетки"""
        if len(self.__channel) > 1: return self.__channel

        Fu = lambda x: self.__fu(x) - self.__relative_step
        step = self.properties['len_l'] / self.__discreteness  # шаг вдоль кривой

        xgmin, xgmax = 0 + self.__relative_inlet_radius, 1 - self.__relative_outlet_radius
        if isnan(xgmin): xgmin = 0
        if isnan(xgmax): xgmin = 1

        x = [xgmin]
        while True:
            X = x[-1] + step * tan2cos(derivative(self.__fl, x[-1]))
            if X > xgmax: break
            x.append(X)
        x = array(x + [xgmax], dtype='float64')

        Au, _, Cu = coefficients_line(func=self.__fl, x0=x)

        def equations(vars, *args):
            """СНЛАУ"""
            x0, y0, r0, xl = vars
            xu, yu, Au, Cu = args

            Al, _, Cl = coefficients_line(func=Fu, x0=xl)

            return [abs(Au * x0 + (-1) * y0 + Cu) / sqrt(Au ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xu - x0) ** 2 + (yu - y0) ** 2) - r0 ** 2,  # до кривой корыта
                    abs(Al * x0 + (-1) * y0 + Cl) / sqrt(Al ** 2 + 1) - r0,  # расстояние от точки окружности
                    ((xl - x0) ** 2 + (Fu(xl) - y0) ** 2) - r0 ** 2]  # до кривой спинки

        xd, yd, d = list(), list(), list()

        warnings.filterwarnings('error')
        for xu, yu, a_u, c_u in tqdm(zip(x, self.__fl(x), Au, Cu), desc='Channel calculation', total=len(x)):
            try:
                res = fsolve(equations, array((xu, yu, self.__relative_step / 2, xu)), args=(xu, yu, a_u, c_u))
            except Exception:
                continue

            if all((xgmin <= res[0] <= xgmax,
                    Fu(res[0]) < res[1] < self.__fl(res[0]),  # y центра окружности лежит в канале
                    xgmin <= res[3] <= xgmax,
                    res[2] * 2 <= self.__relative_step)):
                xd.append(res[0])
                yd.append(res[1])
                d.append(res[2] * 2)
        warnings.filterwarnings('default')

        r = zeros(len(d), dtype='float64')
        for i in range(1, len(d)): r[i] = r[i - 1] + distance((xd[i - 1], yd[i - 1]), (xd[i], yd[i]))

        self.__channel = array((xd, yd, d, r), dtype='float64').T

        return self.__channel

    # TODO
    def cfd(self, vx, vy, padding=0.2, xlim=None, ylim=None):
        """Продувка"""
        assert isinstance(vx, (int, float, np.number))
        assert isinstance(vy, (int, float, np.number))
        assert isinstance(padding, (float, int)) and 0 <= padding

        def u(xy: tuple, vortexs, bounds: tuple = (1, 1)):
            X, Y = xy
            ux = np.full_like(X, bounds[0], dtype=np.float64)
            uy = np.full_like(Y, bounds[1], dtype=np.float64)

            for i, j, k in tqdm(vortexs, desc='CFD'):
                R = ((X - i) ** 2 + (Y - j) ** 2)
                ux += -k * (Y - j) / R
                uy += k * (X - i) / R

            return ux, uy

        x, y = array(self.coordinates).T
        upper_lower = self.upper_lower(self.coordinates)
        upper, lower = upper_lower['upper'], upper_lower['lower']

        vortexs = array((x, y,
                         [-0.5] * (len(x) // 2) + [0.5] * (len(x) - len(x) // 2),  # np.random.randn(len(x))/3
                         )).T

        if xlim is None:
            width = x.max() - x.min()
            xlim = (x.min() - padding * width, x.max() + padding * width)
        if ylim is None:
            height = y.max() - y.min()
            ylim = (y.min() - padding * height, y.max() + padding * height)

        X, Y = np.meshgrid(linspace(*xlim, self.__discreteness * 2), linspace(*ylim, self.__discreteness ** 2))
        ux, uy = u((X, Y), vortexs, bounds=(vx, vy))
        '''for i, ux_ in enumerate(ux):
            for j, uy_ in enumerate(uy):
                if array(upper_lower['lower']) <= uy <= upper_lower['upper']:
                    ux[i], uy[j] = 0, 0'''

        plt.figure(dpi=150)
        plt.streamplot(X, Y, ux, uy,
                       color=(0, 0, 1, 0.5), density=1.5, minlength=0.1, linewidth=0.8, broken_streamlines=True)
        plt.plot(x, y, color='black', linewidth=2)
        plt.axis('equal')
        plt.show()

    def to_dataframe(self) -> pd.DataFrame:
        """Перевод координат в pandas.DataFrame"""
        return pd.DataFrame(self.__coordinates, columns=('x', 'y'))

    def export(self):
        """Экспортирование координат и характеристик профиля"""
        if not os.path.isdir('datas'): os.mkdir('datas')
        ctime = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.to_dataframe().to_excel(f'datas/airfoil_coordinates_{ctime}.xlsx', header=True)
        pd.DataFrame(self.properties, index=[0]).to_excel(f'datas/airfoil_properties_{ctime}.xlsx', header=True)


def test() -> None:
    """Тестирование"""
    print(Airfoil.__version__)

    Airfoil.help()

    Airfoil.rnd = 4
    print(f'Airfoil.rnd: {Airfoil.rnd}')

    airfoils = list()

    if 1:
        airfoils.append(Airfoil('BMSTU', 30, 1 / 1.698, radians(46.23)))

        airfoils[-1].rotation_angle = radians(70)
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

        airfoils[-1].mynk_coefficient = 0.2

    if 1:
        airfoils.append(Airfoil('PARSEC', 50, 1 / 1.698, radians(46.23)))

        airfoils[-1].relative_inlet_radius = 0.06
        airfoils[-1].x_relative_camber_upper, airfoils[-1].x_relative_camber_lower = 0.35, 0.45
        airfoils[-1].relative_camber_upper, airfoils[-1].relative_camber_lower = 0.055, -0.006
        airfoils[-1].d2y_dx2_upper, airfoils[-1].d2y_dx2_lower = -0.35, -0.2
        airfoils[-1].theta_outlet_upper, airfoils[-1].theta_outlet_lower = radians(-6), radians(0.05)

    if 1:
        airfoils.append(Airfoil('BEZIER', 30, 1 / 1.698, radians(46.23)))

        airfoils[-1].points = ((1.0, 0.0), (0.35, 0.200), (0.05, 0.100),
                               (0.0, 0.0),
                               (0.05, -0.10), (0.35, -0.05), (0.5, 0.0), (1.0, 0.0))

    if 1:
        airfoils.append(Airfoil('MANUAL', 30, 1 / 1.698, radians(46.23)))

        airfoils[-1].upper = ((0.0, 0.0), (0.05, 0.08), (0.10, 0.110), (0.35, 0.150), (0.5, 0.15), (1.0, 0.0))
        airfoils[-1].lower = ((0.0, 0.0), (0.05, -0.025), (0.35, -0.025), (0.5, 0.0), (0.8, 0.025), (1.0, 0.0))
        airfoils[-1].deg = 3

    if 1:
        airfoils.append(Airfoil('CIRCLE', 60, 0.5, radians(30)))

        airfoils[-1].relative_circles = ((0.1, 0.04),
                                         (0.2, 0.035),
                                         (0.3, 0.03),
                                         (0.4, 0.028),
                                         (0.5, 0.025),
                                         (0.6, 0.02),)
        airfoils[-1].rotation_angle = radians(40)
        airfoils[-1].x_ray_cross = 0.5
        airfoils[-1].is_airfoil = True

    if 1:
        airfoils.append(Airfoil('CIRCLE', 60, 0.5, radians(30)))

        airfoils[-1].relative_circles = ((0.1, 0.4),
                                         (0.2, 0.4),
                                         (0.3, 0.4),
                                         (0.4, 0.4),
                                         (0.5, 0.4),
                                         (0.6, 0.4),
                                         (0.8, 0.4),
                                         (0.9, 0.4))
        airfoils[-1].rotation_angle = radians(40)
        airfoils[-1].x_ray_cross = 0.5
        airfoils[-1].is_airfoil = False

    for airfoil in airfoils:
        airfoil.show()

        print(airfoil.to_dataframe())

        print(Fore.MAGENTA + 'airfoil properties:' + Fore.RESET)
        for k, v in airfoil.properties.items(): print(f'{k}: {v}')

        print(Fore.MAGENTA + 'airfoil channel:' + Fore.RESET)
        print(f'{airfoil.channel}')

        airfoil.cfd(10, 5)

        airfoil.export()


if __name__ == '__main__':
    import cProfile

    cProfile.run('test()', sort='cumtime')
