# Generate and plot the contour of an airfoil using the PARSEC parameterization
# H. Sobieczky, *'Parametric airfoils and wings'* in *Notes on Numerical Fluid Mechanics*, Vol. 68, pp 71-88]
# (www.as.dlr.de/hs/h-pdf/H141.pdf)
# Repository & documentation: http://github.com/dqsis/parsec-airfoils


import numpy as np
import matplotlib.pyplot as plt

from curves import parsec_coef

'''
* pressure and suction surface crest locations (x_pre, y_pre, x_suc, y_suc)
* curvatures at the pressure and suction surface crest locations (d2y/dx2_pre, d2y/dx2_suc)
* trailing edge coordinates (x_TE, y_TE)
* trailing edge angles between the pressure and suction surface and the horizontal axis (th_pre, th_suc)
'''

radius_inlet = 0.01  # радиус входной кромки

# Pressure (lower) surface parameters 
x_pre = 0.450
y_pre = -0.006
d2ydx2_pre = -0.2
th_pre = 0.05

# Suction (upper) surface parameters
x_suc = 0.350
y_suc = 0.055
d2ydx2_suc = -0.350
th_suc = -6

# Evaluate pressure (lower) surface coefficients
cf_pre = parsec_coef(1, 0, radius_inlet,
                     x_pre, y_pre, d2ydx2_pre, th_pre,
                     'pre')

# Evaluate suction (upper) surface coefficients
cf_suc = parsec_coef(1, 0, radius_inlet,
                     x_suc, y_suc, d2ydx2_suc, th_suc,
                     'suc')

# Evaluate pressure (lower) surface points
xx_pre = np.linspace(1, 0, 101)
yy_pre = (cf_pre[0] * xx_pre ** (1 / 2) +
          cf_pre[1] * xx_pre ** (3 / 2) +
          cf_pre[2] * xx_pre ** (5 / 2) +
          cf_pre[3] * xx_pre ** (7 / 2) +
          cf_pre[4] * xx_pre ** (9 / 2) +
          cf_pre[5] * xx_pre ** (11 / 2)
          )

# Evaluate suction (upper) surface points
xx_suc = np.linspace(0, 1, 101)
yy_suc = (cf_suc[0] * xx_suc ** (1 / 2) +
          cf_suc[1] * xx_suc ** (3 / 2) +
          cf_suc[2] * xx_suc ** (5 / 2) +
          cf_suc[3] * xx_suc ** (7 / 2) +
          cf_suc[4] * xx_suc ** (9 / 2) +
          cf_suc[5] * xx_suc ** (11 / 2)
          )

# Plot airfoil contour
plt.figure()
plt.plot(xx_suc, yy_suc, 'b', xx_pre, yy_pre, 'r', linewidth=2)
plt.grid(True)
plt.xlim([0, 1])
plt.yticks(np.arange(-0.5, 0.6, 0.1))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.axis('equal')

# plt.title("PARSEC airfoil with parameters:\n{}\n{}".format(parnv[0],parnv[1]))
# Make room for title automatically
plt.tight_layout()
plt.show()
