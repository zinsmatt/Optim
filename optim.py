import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen, rosen_der, rosen_hess

N = 100
B = 2
x = np.linspace(-B, B, N)


def f(x):
    return 0.1 * x**2 -x + np.cos(x) * 10 - 1
def f_prim(x):
    return 2 * 0.1 * x - 1 - 10 * np.sin(x)
def f_second(x):
    return 2*0.1-10*np.cos(x)

y = f(x)

b = 1.0

def g(x):
    return (1/2) * (f(x) - b)**2

def g_prim(x):
    return f_prim(x) * (f(x) - b)
def g_second(x):
    return f_second(x) * f(x) + f_prim(x) * f_prim(x) - b * f_second(x)

G = g(x)

# Newton's method: Iterative where optimal displacement dx is found by
# g_prim(x) + g_second(x) * dx = 0
# dx = -g_prim(x) / g_second(x)
# Warning this is done on the objective function g(x)

print("Newton:")
x0 = 0.8
d = 100
print("x0: ", x0)
x_cur = x0
eps = 1.0e-5
iterations = [x0]
iter = 1
while abs(d) > eps:
    # print("f_prim(x) = ", g_prim(x_cur))
    # print("f_second(x) = ", g_second(x_cur))
    d = -g_prim(x_cur) / g_second(x_cur)
    # print("d = ", d)
    x_cur += d
    print("Iter", iter, "=", x_cur)
    iterations.append(x_cur)
    iter += 1

iterations = np.array(iterations)
print(x_cur)

print("Final g = ", g(x_cur))

plt.scatter(x, y, label="f")
plt.scatter(x, G, label="g")

plt.scatter(iterations, g(iterations))
plt.legend()
plt.grid()
plt.show()
