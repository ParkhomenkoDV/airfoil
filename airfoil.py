import sys
from math import nan, inf, pi, radians, degrees, sqrt, sin, cos, tan, atan, floor, ceil
from numpy import linspace, arange, zeros
from scipy import interpolate, integrate
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl

from colorama import Fore
from curves import BernsteinCurve

sys.path.append('D:/Programming/Python')

from tools import export2, isnum, COOR, Axis, angle, rounding, eps, dist, dist2line
from decorators import timeit

import cProfile
from numba import njit


class Airfoil:
    """Относительный аэродинамический профиль"""

    __Nmin, __Nrecomend = 3, 20

    def __init__(self, method: str, N: int = __Nrecomend):
        self.__method = method  # метод построения аэродинамического профиля
        self.__N = N  # количество точек дискретизации
        self.coords = dict()
        self.__props = dict()

    def __str__(self) -> str:
        return self.__method

    @property
    def method(self) -> str:
        return self.__method

    @method.setter
    def method(self, method):
        self.reset()
        if type(method) is str and method:
            self.__method = method.strip()
        else:
            print(Fore.RED + f'method is str in []!')

    @method.deleter
    def method(self):
        self.reset()

    @property
    def N(self) -> int:
        return self.__N

    @N.setter
    def N(self, N):
        self.reset()
        if type(N) is int and N >= 3:
            self.__N = N
        else:
            print(Fore.RED + f'N is int >= 3!')

    @N.deleter
    def N(self):
        self.reset()

    def reset(self):
        self.__method = ''
        self.__N = self.__Nrecomend
        self.coords = dict()
        self.__props = dict()

    def input(self):
        pass

    def validate(self) -> bool:
        if self.method.upper() in ('NACA', 'N.A.C.A.'):
            pass
        if self.method.upper() in ('BMSTU', 'МГТУ', 'МВТУ', 'МИХАЛЬЦЕВ'):
            # относ. координата максимальной выпуклости профиля
            assert hasattr(self, 'xg_b') and self.xg_b and type(self.xg_b) is float and 0 <= self.xg_b <= 1, 'xg_b'

            # относ. радиус входной и выходной кромок
            assert (hasattr(self, 'r_inlet_b') and self.r_inlet_b and
                    type(self.r_inlet_b) is float and 0 <= self.r_inlet_b <= 1), 'r_inlet_b'
            assert (hasattr(self, 'r_outlet_b') and self.r_outlet_b and
                    type(self.r_outlet_b) is float and 0 <= self.r_outlet_b <= 1), 'r_outlet_b'

            # степень приближенности к спинке
            assert hasattr(self, 'g_') and type(self.g_) is float and 0 <= self.g_ <= 1, 'g_'

            # угол раскрытия входной и выходной кромок
            assert hasattr(self, 'g_inlet') and type(self.g_inlet) is float and 0 <= self.g_inlet, 'g_inlet'
            assert hasattr(self, 'g_outlet') and type(self.g_outlet) is float and 0 <= self.g_outlet, 'g_outlet'

            # угол поворота потока
            assert hasattr(self, 'e') and self.e and type(self.e) is float, 'e'

        if self.method.upper() in ('BEZIER', 'БЕЗЬЕ'):
            pass
        return True

    def BMSTU(self):
        self.validate()
        self.coords['u'], self.coords['d'] = {'x': [], 'y': []}, {'x': [], 'y': []}

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

        self.__k_inlet, self.__k_outlet = k_inlet, k_outlet

        # углы входа и выхода профиля
        if self.e > 0:
            g_u_inlet, g_d_inlet = (1 - self.g_) * self.g_inlet, self.g_ * self.g_inlet
            g_u_outlet, g_d_outlet = (1 - self.g_) * self.g_outlet, self.g_ * self.g_outlet
        else:
            g_u_inlet, g_d_inlet = self.g_ * self.g_inlet, (1 - self.g_) * self.g_inlet,
            g_u_outlet, g_d_outlet = self.g_ * self.g_outlet, (1 - self.g_) * self.g_outlet

        # положения центров окружностей входной и выходной кромок
        xo_inlet, yo_inlet = self.r_inlet_b, k_inlet * self.r_inlet_b
        xo_outlet, yo_outlet = 1 - self.r_outlet_b, -k_outlet * self.r_outlet_b

        # точки пересечения линий спинки и корыта
        xcl_u, ycl_u = COOR(tan(atan(k_inlet) + g_u_inlet),
                            sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * self.r_inlet_b -
                            (tan(atan(k_inlet) + g_u_inlet)) * xo_inlet - (-1) * yo_inlet,
                            tan(atan(k_outlet) - g_u_outlet),
                            sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * self.r_outlet_b -
                            (tan(atan(k_outlet) - g_u_outlet)) * xo_outlet - (-1) * yo_outlet)

        xcl_d, ycl_d = COOR(tan(atan(k_inlet) - g_d_inlet),
                            -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * self.r_inlet_b -
                            (tan(atan(k_inlet) - g_d_inlet)) * xo_inlet - (-1) * yo_inlet,
                            tan(atan(k_outlet) + g_d_outlet),
                            -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * self.r_outlet_b -
                            (tan(atan(k_outlet) + g_d_outlet)) * xo_outlet - (-1) * yo_outlet)

        # точки пересечения окружностей со спинкой и корытом
        xclc_i_u, yclc_i_u = COOR(tan(atan(k_inlet) + g_u_inlet),
                                  sqrt(tan(atan(k_inlet) + g_u_inlet) ** 2 + 1) * self.r_inlet_b
                                  - (tan(atan(k_inlet) + g_u_inlet)) * xo_inlet - (-1) * yo_inlet,
                                  -1 / (tan(atan(k_inlet) + g_u_inlet)),
                                  -(-1 / tan(atan(k_inlet) + g_u_inlet)) * xo_inlet - (-1) * yo_inlet)

        xclc_i_d, yclc_i_d = COOR(tan(atan(k_inlet) - g_d_inlet),
                                  -sqrt(tan(atan(k_inlet) - g_d_inlet) ** 2 + 1) * self.r_inlet_b
                                  - (tan(atan(k_inlet) - g_d_inlet)) * xo_inlet - (-1) * yo_inlet,
                                  -1 / (tan(atan(k_inlet) - g_d_inlet)),
                                  -(-1 / tan(atan(k_inlet) - g_d_inlet)) * xo_inlet - (-1) * yo_inlet)

        xclc_e_u, yclc_e_u = COOR(tan(atan(k_outlet) - g_u_outlet),
                                  sqrt(tan(atan(k_outlet) - g_u_outlet) ** 2 + 1) * self.r_outlet_b
                                  - tan(atan(k_outlet) - g_u_outlet) * xo_outlet - (-1) * yo_outlet,
                                  -1 / tan(atan(k_outlet) - g_u_outlet),
                                  -(-1 / tan(atan(k_outlet) - g_u_outlet)) * xo_outlet - (-1) * yo_outlet)

        xclc_e_d, yclc_e_d = COOR(tan(atan(k_outlet) + g_d_outlet),
                                  -sqrt(tan(atan(k_outlet) + g_d_outlet) ** 2 + 1) * self.r_outlet_b
                                  - tan(atan(k_outlet) + g_d_outlet) * xo_outlet - (-1) * yo_outlet,
                                  -1 / tan(atan(k_outlet) + g_d_outlet),
                                  -(-1 / tan(atan(k_outlet) + g_d_outlet)) * xo_outlet - (-1) * yo_outlet)

        # точки входной окружности кромки по спинке
        an = angle(points=((0, yo_inlet), (xo_inlet, yo_inlet), (xclc_i_u, yclc_i_u)))
        if xclc_i_u > xo_inlet: an = pi - an
        for a in arange(0, an, an / self.__N):
            self.coords['u']['x'].append(self.r_inlet_b * (1 - cos(a)))
            self.coords['u']['y'].append(yo_inlet + self.r_inlet_b * sin(a))

        # точки входной окружности кромки по корыту
        an = angle(points=((0, yo_inlet), (xo_inlet, yo_inlet), (xclc_i_d, yclc_i_d)))
        if xclc_i_d > xo_inlet: an = pi - an
        for a in arange(0, an, an / self.__N):
            self.coords['d']['x'].append(self.r_inlet_b * (1 - cos(a)))
            self.coords['d']['y'].append(yo_inlet - self.r_inlet_b * sin(a))

        xu, yu = BernsteinCurve(((xclc_i_u, yclc_i_u), (xcl_u, ycl_u), (xclc_e_u, yclc_e_u)), N=self.__N).T.tolist()
        xd, yd = BernsteinCurve(((xclc_i_d, yclc_i_d), (xcl_d, ycl_d), (xclc_e_d, yclc_e_d)), N=self.__N).T.tolist()
        self.coords['u']['x'] += xu
        self.coords['u']['y'] += yu
        self.coords['d']['x'] += xd
        self.coords['d']['y'] += yd

        an = angle(points=((1, yo_outlet), (xo_outlet, yo_outlet), (xclc_e_u, yclc_e_u)))
        if xo_outlet > xclc_e_u: an = pi - an
        for a in arange(0, an, an / self.__N):
            self.coords['u']['x'].insert(2 * self.__N, 1 - self.r_outlet_b * (1 - cos(a)))
            self.coords['u']['y'].insert(2 * self.__N, yo_outlet + self.r_outlet_b * sin(a))

        an = angle(points=((1, yo_outlet), (xo_outlet, yo_outlet), (xclc_e_d, yclc_e_d)))
        if xo_outlet > xclc_e_d: an = pi - an
        for a in arange(0, an, an / self.__N):
            self.coords['d']['x'].insert(2 * self.__N, 1 - self.r_outlet_b * (1 - cos(a)))
            self.coords['d']['y'].insert(2 * self.__N, yo_outlet - self.r_outlet_b * sin(a))

    def NACA(self):
        pass

    def Bezier(self):
        pass

    @timeit()
    def solve(self):
        if self.method.upper() in ('NACA', 'N.A.C.A.'):
            return self.NACA()
        elif self.method.upper() in ('BMSTU', 'МГТУ', 'МВТУ', 'МИХАЛЬЦЕВ'):
            return self.BMSTU()
        elif self.method.upper() in ('BEZIER', 'БЕЗЬЕ'):
            return self.Bezier()
        else:
            raise 'method not found!'

    def transform(self, x0=0, y0=0, angle=0, scale=1, inplace=False) -> dict:
        """Перенос-поворот кривых спинки и корыта профиля"""

        xun, yun = [nan] * len(self.coords['u']['x']), [nan] * len(self.coords['u']['x'])
        xdn, ydn = [nan] * len(self.coords['d']['x']), [nan] * len(self.coords['d']['x'])

        for i in range(len(self.coords['u']['x'])):
            xun[i], yun[i] = Axis.transform(self.coords['u']['x'][i], self.coords['u']['y'][i],
                                            x0, y0, angle, scale)

        for i in range(len(self.coords['d']['x'])):
            xdn[i], ydn[i] = Axis.transform(self.coords['d']['x'][i], self.coords['d']['y'][i],
                                            x0, y0, angle, scale)

        if inplace:
            self.coords['u']['x'], self.coords['u']['y'] = xun, yun
            self.coords['d']['x'], self.coords['d']['y'] = xdn, ydn

        return {'u': {'x': xun, 'y': yun},
                'd': {'x': xdn, 'y': ydn}}

    def show(self, what='all'):
        """Построение профиля"""
        fg = plt.figure(figsize=(13, 6))  # размер в дюймах
        gs = fg.add_gridspec(1, 3)  # строки, столбцы

        fg.add_subplot(gs[0, 0])  # позиция графика
        plt.title('Initial data')
        plt.grid(False)  # сетка
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(-0.5, 0.5)
        ytemp = 0.35
        plt.text(0.1, ytemp, f'method: {self.method}', fontsize=10)
        '''
        for key in initial_data:
            plt.text(0.1, ytemp, key + ' = ' + str(initial_data[key]), fontsize=10)
            ytemp -= 0.1
        del ytemp'''

        fg.add_subplot(gs[0, 1])  # позиция графика
        plt.title('AIRFOIL with extra structure')
        plt.grid(True)  # сетка
        plt.axis('equal')
        plt.xlim(0, 1)

        plt.plot([0, self.xg_b, 1], [0, self.__k_inlet * self.xg_b, 0],
                 linestyle='--', color='green', linewidth=1.5)

        # plt.plot([xclc_i_u, xcl_u, xclc_e_u], [yclc_i_u, ycl_u, yclc_e_u], 'bo--')
        # plt.plot([xclc_i_d, xcl_d, xclc_e_d], [yclc_i_d, ycl_d, yclc_e_d], 'ro--')

        plt.plot(self.coords['u']['x'], self.coords['u']['y'], ls='-', color='blue', linewidth=2)
        plt.plot(self.coords['d']['x'], self.coords['d']['y'], ls='-', color='red', linewidth=2)
        x_inlet, y_inlet, x_outlet, y_outlet = [], [], [], []
        for alpha in linspace(0, 2 * pi, 360):
            x_inlet.append(self.r_inlet_b * cos(alpha) + self.r_inlet_b)
            y_inlet.append(self.r_inlet_b * sin(alpha) + self.r_inlet_b * self.__k_inlet)
            x_outlet.append(self.r_outlet_b * cos(alpha) + (1 - self.r_outlet_b))
            y_outlet.append(self.r_outlet_b * sin(alpha) - self.r_outlet_b * self.__k_outlet)
        plt.plot(x_inlet, y_inlet, ls='-', color='black', linewidth=1)
        plt.plot(x_outlet, y_outlet, ls='-', color='black', linewidth=1)

        fg.add_subplot(gs[0, 2])  # позиция графика
        plt.title('AIRFOIL')
        plt.grid(True)  # сетка
        plt.axis('equal')
        plt.xlim(0, 1)

        plt.plot(self.coords['u']['x'], self.coords['u']['y'], ls='-', color='black', linewidth=2)
        plt.plot(self.coords['d']['x'], self.coords['d']['y'], ls='-', color='black', linewidth=2)
        plt.show()

    def to_dataframe(self, bears='pandas'):
        if bears.strip().lower() == 'pandas':
            return pd.DataFrame({'xu': airfoil.coords['u']['x'], 'yu': airfoil.coords['u']['y'],
                                 'xd': airfoil.coords['d']['x'], 'yd': airfoil.coords['d']['y']})
        if bears.strip().lower() == 'polars':
            return pl.DataFrame({'xu': airfoil.coords['u']['x'], 'yu': airfoil.coords['u']['y'],
                                 'xd': airfoil.coords['d']['x'], 'yd': airfoil.coords['d']['y']})
        else:
            print(Fore.RED + 'Unknown bears!' + Fore.RESET)
            print('Use "pandas" or "polars"!')

    @property
    @timeit()
    def properties(self, epsrel=1e-4) -> dict:
        if self.__props: return self.__props

        Yup = interpolate.interp1d(*self.coords['u'].values(), kind='quadratic', fill_value='extrapolate')
        Ydown = interpolate.interp1d(*self.coords['d'].values(), kind='quadratic', fill_value='extrapolate')

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
        self.__props['alpha'] = 0.5 * atan(-2 * self.__props['Jxcyc'] / (self.__props['Jxc'] - self.__props['Jyc']))

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

    def __init__(self, airfoil, gamma: float, t_b: float = 1.0, N: int = 20):
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
        if (type(t_b) is float or type(t_b) is int) and t_b > 0:
            self.__t_b = t_b
        else:
            print(Fore.RED + f't_b is float or int > 0!')

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
            print(Fore.RED + f'gamma is float or int!')

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
        self.coords['u'], self.coords['d'] = {'x': [], 'y': []}, {'x': [], 'y': []}

        dct = self.__airfoil.transform(self.__airfoil.properties['x0'], self.__airfoil.properties['y0'],
                                       self.gamma, scale=1, inplace=False)
        XuG, YuG, XdG, YdG = dct['u']['x'], dct['u']['y'], dct['d']['x'], dct['d']['y']
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
        self.coords['d']['x'], self.coords['d']['y'] = XdGt, YdGt

        return self.coords

    @property
    @timeit()
    def properties(self, epsrel=0.01):
        """Дифузорность/конфузорность решетки"""
        if self.__props: return self.__props

        '''d, xd, yd = [nan] * len(XdGt), [nan] * len(XdGt), [nan] * len(XdGt)  # обnanивание списков
    
        for i in range(len(XdGt)):
            if XdGt[i] < min(XdGt) + self.__airfoil.r_inlet_b or XdGt[i] > max(XdGt) - self.__airfoil.r_outlet_b: 
                continue
            d[i] = inf
            for j in range(len(XuGt)):
                if sqrt((XdGt[i] - XuGt[j]) ** 2 + (((YdGt[i] + self.__t_b / 2) - (YuGt[j] - self.__t_b / 2)) ** 2)) < \
                        d[i]:
                    d[i] = sqrt(
                        (XdGt[i] - XuGt[j]) ** 2 + (((YdGt[i] + self.__t_b / 2) - (YuGt[j] - self.__t_b / 2)) ** 2))
                    xd[i] = 0.5 * (XdGt[i] + XuGt[j])
                    yd[i] = 0.5 * ((YdGt[i] + self.__t_b / 2) + (YuGt[j] - self.__t_b / 2))
    
        dt, rd, xdt, ydt = list(), list(), list(), list()
    
        for i in range(len(d)):
            if isnum(str(d[i])) and d[i] is not nan: dt.append(d[i])
            if isnum(str(xd[i])) and xd[i] is not nan: xdt.append(xd[i])
            if isnum(str(yd[i])) and yd[i] is not nan: ydt.append(yd[i])
    
        rd.append(xd[0])
    
        for i in range(1, len(d)):
            if isnum(str(sqrt((xd[i] - xd[i - 1]) ** 2 + (yd[i] - yd[i - 1]) ** 2))):
                rd.append(rd[i - 1] + sqrt((xd[i] - xd[i - 1]) ** 2 + (yd[i] - yd[i - 1]) ** 2))
            else:
                rd.append(0)
    
        self.d, self.rd, self.xd, self.yd = dt, rd, xdt, ydt
        '''

        # kind='cubic' необходим для гладкости производной
        Fd = interpolate.interp1d(self.coords['d']['x'], [y + self.__t_b / 2 for y in self.coords['d']['y']],
                                  kind='cubic', fill_value='extrapolate')
        Fu = interpolate.interp1d(self.coords['u']['x'], [y - self.__t_b / 2 for y in self.coords['u']['y']],
                                  kind='cubic', fill_value='extrapolate')

        xgmin = min(self.coords['u']['x'] + self.coords['d']['x']) + self.__airfoil.r_inlet_b
        ygmin = min(self.coords['d']['y']) - self.__t_b / 2
        xgmax = max(self.coords['u']['x'] + self.coords['d']['x']) - self.__airfoil.r_outlet_b
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
        for i in range(len(Ld)):
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

        self.r = zeros(len(self.d))
        for i in range(1, len(self.d)):
            self.r[i] = self.r[i - 1] + dist((self.xd[i - 1], self.yd[i - 1]), (self.xd[i], self.yd[i]))

        self.__props = {tuple((self.xd[i], self.yd[i])): self.d[i] for i in range(len(self.d))}

        return self.__props

    def show(self):
        if not self.__props: self.properties
        fg = plt.figure(figsize=(12, 9))  # размер в дюймах
        gs = fg.add_gridspec(1, 2)  # строки, столбца

        fg.add_subplot(gs[0, 0])  # позиция графика
        plt.title('Решетка')
        plt.grid(True)  # сетка
        plt.axis('square')
        plt.xlim(floor(min(self.coords['u']['x'] + self.coords['d']['x'])),
                 ceil(max(self.coords['u']['x'] + self.coords['d']['x'])))
        plt.ylim(-1, 1)
        for i in range(len(self.d)):
            plt.plot(list(self.d[i] / 2 * cos(alpha) + self.xd[i] for alpha in linspace(0, 2 * pi, 360)),
                     list(self.d[i] / 2 * sin(alpha) + self.yd[i] for alpha in linspace(0, 2 * pi, 360)),
                     ls='-', color=(0, 1, 0))
        plt.plot(self.xd, self.yd, ls='--', color=(1, 0, 1))
        plt.plot([min(self.coords['u']['x'] + self.coords['d']['x'])] * 2, [-1, 1],
                 [max(self.coords['u']['x'] + self.coords['d']['x'])] * 2, [-1, 1],
                 ls='-', color=(0, 0, 0))  # границы решетки
        plt.plot(self.coords['u']['x'], list(y - self.__t_b / 2 for y in self.coords['u']['y']),
                 ls='-', color=(0, 0, 0))
        plt.plot(self.coords['d']['x'], list(y - self.__t_b / 2 for y in self.coords['d']['y']),
                 ls='-', color=(0, 0, 0))
        plt.plot(self.coords['u']['x'], list(y + self.__t_b / 2 for y in self.coords['u']['y']),
                 ls='-', color=(0, 0, 0))
        plt.plot(self.coords['d']['x'], list(y + self.__t_b / 2 for y in self.coords['d']['y']),
                 ls='-', color=(0, 0, 0))

        fg.add_subplot(gs[0, 1])  # позиция графика
        plt.title('Канал')
        plt.grid(True)  # сетка
        plt.axis('square')
        plt.xlim(0, ceil(max(self.r)))
        plt.ylim(0, ceil(self.__t_b))
        plt.plot(self.r, self.d, ls='-', color=(0, 1, 0))
        plt.plot([0, ceil(max(self.r))], [0, 0], ls='-', color=(0, 0, 0), linewidth=1.5)
        plt.plot(list((self.r[i] + self.r[i + 1]) / 2 for i in range(len(self.r) - 1)),
                 list((self.d[i + 1] - self.d[i]) / (self.r[i + 1] - self.r[i]) for i in range(len(self.r) - 1)),
                 ls='-', color=(1, 0, 0))
        plt.show()


def airfoil_properties(Xu, Yu, Xd, Yd):
    print('Airfoil properties:')
    for key in prop: print(key, ' = ', prop[key])

    fg = plt.figure(figsize=(13, 6))  # размер в дюймах
    gs = fg.add_gridspec(1, 2)  # строки, столбца

    fg.add_subplot(gs[0, 0])  # позиция графика
    plt.title('AIRFOIL properties', fontsize=14)
    plt.grid(True)  # сетка
    plt.axis('square')
    plt.xlim(0, 1)
    if initial_data['e'] == 0:
        plt.ylim(-0.5, 0.5)
    elif 0 < degrees(initial_data['e']):
        plt.ylim(-0.25, 0.75)
    else:
        plt.ylim(-0.75, 0.25)
    x = list(linspace(0, initial_data['r_вх'], len(Xu))) + \
        list(linspace(initial_data['r_вх'], 1 - initial_data['r_вых'], len(Xu))) + \
        list(linspace(1 - initial_data['r_вых'], 1, len(Xu)))
    plt.plot(Xu, Yu, ls='-', color='black', linewidth=2)
    plt.plot(Xd, Yd, ls='-', color='black', linewidth=2)
    plt.plot(x, [tan(prop['alpha']) * (x - prop['xc']) + prop['yc'] for x in x], 'r--')
    plt.plot(x, [tan(prop['alpha'] + pi / 2) * (x - prop['xc']) + prop['yc'] for x in x], 'r--')
    plt.text(prop['xc'], prop['yc'], 'C', fontsize=14, fontweight='bold')

    fg.add_subplot(gs[0, 1])  # позиция графика
    plt.title('AIRFOIL properties', fontsize=14)
    plt.grid(False)  # сетка
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    ytemp = 0.4
    for key in prop:
        plt.text(0.1, ytemp, key + ' = ' + str(prop[key]))
        ytemp -= 0.055
        if key == ['alpha']: plt.text(0.1, ytemp, key + ' = ' + str(degrees(properties[key])))
    del ytemp
    plt.show()

    return prop


def show(show='all', savefig=False):
    plt.close('all')

    fg = plt.figure(figsize=(13, 9))  # размер в дюймах
    gs = fg.add_gridspec(2, 3)  # строки, столбца

    fg.add_subplot(gs[0, 0])  # позиция графика
    plt.title('Initial data (Исходные данные)', fontsize=12)
    plt.grid(False)  # сетка
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    ytemp = 0.35
    for key in initial_data:
        if key in ('g_вх', 'g_вых', 'e'):
            plt.text(0.1, ytemp, key +
                     ' = ' + str(rounding(initial_data[key], rnd)) +
                     ' = ' + str(rounding(degrees(initial_data[key]), rnd)) + ' deg', fontsize=10)
        else:
            plt.text(0.1, ytemp, key + ' = ' + str(rounding(initial_data[key], rnd)), fontsize=10)
        ytemp -= 0.1
    del ytemp

    fg.add_subplot(gs[0, 1])  # позиция графика
    plt.title('AIRFOIL (профиль)', fontsize=12)
    plt.grid(True)  # сетка
    plt.axis('scaled')
    plt.xlim(0, 1)
    if initial_data['e'] == 0:
        plt.ylim(-0.5, 0.5)
    elif 0 < degrees(initial_data['e']):
        plt.ylim(-0.25, 0.75)
    else:
        plt.ylim(-0.75, 0.25)
    plt.plot(Xu, Yu, Xd, Yd, '-', color='black', linewidth=2)

    fg.add_subplot(gs[1, 0])  # позиция графика
    plt.title('Properties (хар-ки)', fontsize=12)
    plt.grid(False)  # сетка
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(-0.5, 0.5)
    ytemp = 0.425
    for key in properties:
        if key == 'alpha':
            plt.text(0.1, ytemp, key +
                     ' = ' + str(rounding(properties[key], rnd)) +
                     ' = ' + str(rounding(degrees(properties[key]), rnd)) + ' deg')
        else:
            plt.text(0.1, ytemp, key + ' = ' + str(rounding(properties[key], rnd)))
        ytemp -= 0.055
    del ytemp

    fg.add_subplot(gs[1, 1])  # позиция графика
    plt.title('AIRFOIL properties', fontsize=14)
    plt.grid(True)  # сетка
    plt.axis('square')
    plt.xlim(0, 1)
    if initial_data['e'] == 0:
        plt.ylim(-0.5, 0.5)
    elif 0 < degrees(initial_data['e']):
        plt.ylim(-0.25, 0.75)
    else:
        plt.ylim(-0.75, 0.25)
    x = list(linspace(0, initial_data['r_вх'], len(Xu))) + \
        list(linspace(initial_data['r_вх'], 1 - initial_data['r_вых'], len(Xu))) + \
        list(linspace(1 - initial_data['r_вых'], 1, len(Xu)))
    plt.plot(Xu, Yu, ls='-', color='black', linewidth=2)
    plt.plot(Xd, Yd, ls='-', color='black', linewidth=2)
    plt.plot(x, list(tan(properties['alpha']) * (x - properties['xc']) + properties['yc'] for x in x), 'r--')
    plt.plot(x, list(tan(properties['alpha'] + pi / 2) * (x - properties['xc']) + properties['yc'] for x in x), 'r--')
    plt.text(properties['xc'], properties['yc'], 'C', fontsize=14, fontweight='bold')

    fg.add_subplot(gs[0, 2])  # позиция графика
    plt.title('Решетка', fontsize=14)
    plt.grid(True)  # сетка
    plt.axis('square')
    plt.xlim(min(XuG + XdG), max(XuG + XdG))
    plt.ylim(-0.5 - t / 2, 0.5 + t / 2)
    plt.plot([min(XuG + XdG), min(XuG + XdG)], [-1 - t / 2, 1 + t / 2],
             [max(XuG + XdG), max(XuG + XdG)], [-1 - t / 2, 1 + t / 2],
             ls='-', color=(0, 0, 0))  # границы решетки
    for i in range(len(d)):
        plt.plot(list(d[i] / 2 * cos(alpha) + xd[i] for alpha in linspace(0, 2 * pi, 360)),
                 list(d[i] / 2 * sin(alpha) + yd[i] for alpha in linspace(0, 2 * pi, 360)),
                 ls='-', color=(0, 1, 0))
    plt.plot(XuG, list(YuG[i] - t / 2 for i in range(len(YuG))), ls='-', color=(0, 0, 0))
    plt.plot(XdG, list(YdG[i] - t / 2 for i in range(len(YdG))), ls='-', color=(0, 0, 0))
    plt.plot(XuG, list(YuG[i] + t / 2 for i in range(len(YuG))), ls='-', color=(0, 0, 0))
    plt.plot(XdG, list(YdG[i] + t / 2 for i in range(len(YdG))), ls='-', color=(0, 0, 0))
    plt.plot(xd, yd, ls='--', color=(1, 0, 1))

    fg.add_subplot(gs[1, 2])  # позиция графика
    plt.title('Канал', fontsize=14)
    plt.grid(True)  # сетка
    plt.axis('square')
    plt.xlim(min(XuG + XdG), max(XuG + XdG))
    plt.ylim(-0.25, t + 0.25)
    plt.plot([min(XuG + XdG), min(XuG + XdG)], [-1, 1],
             [max(XuG + XdG), max(XuG + XdG)], [-1, 1],
             ls='-', color=(0, 0, 0))
    plt.plot(rd, d, ls='-', color=(0, 1, 0))
    plt.plot([-1, 2], [0, 0], ls='-', color=(0, 0, 0), linewidth=1.5)
    plt.plot(list(0.5 * (rd[i] + rd[i + 1]) for i in range(len(rd) - 1)),
             list((d[i + 1] - d[i]) / (rd[i + 1] - rd[i]) for i in range(len(rd) - 1)),
             ls='-', color=(1, 0, 0))

    # plt.draw()
    # plt.pause(1)
    # plt.waitforbuttonpress()
    export2file(plt, file_path='exports/airfoil',
                file_name='airfoil' +
                          '_' + 'xf' + str(rounding(initial_data['xf'], rnd)) +
                          '_' + 'r_вх' + str(rounding(initial_data['r_вх'], rnd)) +
                          '_' + 'r_вых' + str(rounding(initial_data['r_вых'], rnd)) +
                          '_' + 'g' + str(rounding(initial_data['g'], rnd)) +
                          '_' + 'g_вх' + str(rounding(degrees(initial_data['g_вх']), rnd)) +
                          '_' + 'g_вых' + str(rounding(degrees(initial_data['g_вых']), rnd)) +
                          '_' + 'e' + str(rounding(degrees(initial_data['e']), rnd)),
                file_extension='png', show_time=False)
    plt.show()


if __name__ == '__main__':
    rnd = 4  # количество значащих цифр

    airfoil = Airfoil('BMSTU', 20)  # степень дискретизации

    airfoil.xg_b = 0.35  # относ. координата смены входа/выхода
    # относ. радиус входной и выходной кромок
    airfoil.r_inlet_b = 0.06
    airfoil.r_outlet_b = 0.03
    airfoil.g_ = 0.5  # степень приближенности к спинке
    # угол раскрытия входной и выходной кромок
    airfoil.g_inlet = radians(25)
    airfoil.g_outlet = radians(10)
    airfoil.e = radians(110)  # угол поворота потока

    airfoil.solve()
    airfoil.show()

    print(airfoil.to_dataframe(bears='pandas'))
    # print(airfoil.to_dataframe(bears='polars'))

    print('\nairfoil properties:')
    for k, v in airfoil.properties.items(): print(f'{k}: {v}')

    airfoil.export()

    grate = Grate(airfoil, radians(40), 0.8, N=20)  # относ. шаг профиля, угол установки профиля

    grate.solve()
    grate.show()

    # print(grate.to_dataframe())
    # print(grate.to_dataframe(bears='polars'))

    for k, v in grate.properties.items(): print(f'{k}: {v}')
