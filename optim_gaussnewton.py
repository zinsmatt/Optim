import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen, rosen_der, rosen_hess

N = 100
B = 2.0
x_space = np.linspace(-B, B, N)

# residuals = g(x)-y
y = 7

a = 2.0
b = 1
c = 1.0

def r(x):
    return (a-x)**2 + b*(c-x**2)**2 - y
def r_prim(x):
    return -2*a+2*x-4*b*c*x + 4*b*x**3
def r_second(x):
    return 2-4*b*c+12*b*x**2

residuals = r(x_space)

# y = f(x)

# b = 1.0


# f is the cost function: (1/2) Sum of square residuals (0.5*sum(r(x)**2))
def f(x):
    return 1/2 * r(x)**2
def f_prim(x):
    return r_prim(x) * r(x)
def f_second(x):
    return r_second(x) * r(x) + r_prim(x) * r_prim(x)


# def g(x):
#     return (1/2) * (f(x) - b)**2

# def g_prim(x):
#     return f_prim(x) * (f(x) - b)
# def g_second(x):
#     return f_second(x) * f(x) + f_prim(x) * f_prim(x) - b * f_second(x)

# G = g(x)

# # Gauss-Newton's method: 
# Two interpretations:
#   1. Approximate second derivative
#   2. Linearize the residuals around the current point



# 2. Linearize r around the current point r_: r(x_ + d) = r(x_) + r'(x_) * d
# cost = (1/2) * (r(x_) + r'(x_) * d)^2
# the goal is to find the best value for d. This is a linear case (solve 1/2*(ax-b)^2 )
# solution => d = -r(x_) / r'(x)


print("Newton:")
x0 = 0.8
d = 100
print("x0: ", x0)
x_cur = x0
eps = 1.0e-5
iterations = [x0]
iter = 1
while abs(d) > eps:
    d = -r(x_cur) / r_prim(x_cur)
    x_cur += d
    print("Iter", iter, "=", x_cur)
    iterations.append(x_cur)
    iter += 1

iterations = np.array(iterations)
print(x_cur)

print("Final cost = ", f(x_cur))

plt.scatter(x_space, residuals, label="residual")
plt.scatter(x_space, f(x_space), label="cost function")
plt.scatter(iterations, f(iterations))
plt.legend()
plt.ylim([-5, 50])
plt.grid()
plt.show()
