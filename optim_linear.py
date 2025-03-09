import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen, rosen_der, rosen_hess

N = 100
B = 5
x = np.linspace(-B, B, N)


# def f(x):
#     return 0.1 * x**3 -x + np.cos(x) * 10 - 1

def f(x):
    return 0.5*x

y = f(x)

b = 1.0

G = (1/2) * (f(x) - b)**2

# f = a*x-b

# (1/2)(f(x)-b)(f(x)-b) = 0
# (1/2)f(x)**2 -2*f(x)*b + b = 0
# (1/2)2*f_prim(x)f(x)-2*b = 0
# f_prim(x)f(x)-b = 0

# a * a * x -  b = 0
# x = b / a**2

# (Ax-b)(Ax-b)
# A**2 x**2 -2 A x b + b
a=0.5
GG = (a**2*x**2-2*a*x*b+b)/2

# derivative = 0 
# a**2*x-a*b = 0
# a*a*x = a*b
# x = b/a

print("min in x = ", b / a)

plt.scatter(x, y)
plt.scatter(x, G)
plt.scatter(x, GG)
plt.grid()
plt.show()
