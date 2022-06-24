import numpy as np


def Symplectic4(x0, y0, h, f1, f2):
    gamma1 = 1 / (2 - np.power(2, 1 / 3))
    gamma2 = 1 - 2 * gamma1
    x, y = x0, y0
    x, y = Verlet(x, y, h * gamma1, f1, f2)
    x, y = Verlet(x, y, h * gamma2, f1, f2)
    x, y = Verlet(x, y, h * gamma1, f1, f2)
    return [x, y]


def Verlet(x0, y0, h, f1, f2):
    retx = x0 + f1(y0) * h / 2.0
    rety = y0 + f2(retx) * h
    retx = retx + f1(rety) * h / 2.0
    return [retx, rety]


def SymplecticEuler(x0, y0, h, f1, f2):
    retx = x0 + h * f1(y0)
    rety = y0 + h * f2(retx)
    return [retx, rety]


def RK4(x0, y0, h, f1, f2):
    kx1, ky1 = f1(x0, y0), f2(x0, y0)
    kx2, ky2 = f1(x0 + h * kx1 / 2,
                  y0 + h * ky1 / 2), f2(x0 + h * kx1 / 2, y0 + h * ky1 / 2)
    kx3, ky3 = f1(x0 + h * kx2 / 2,
                  y0 + h * ky2 / 2), f2(x0 + h * kx2 / 2, y0 + h * ky2 / 2)
    kx4, ky4 = f1(x0 + h * kx3, y0 + h * ky3), f2(x0 + h * kx3, y0 + h * ky3)
    return x0 + (kx1 / 6 + kx2 / 3 + kx3 / 3 +
                 kx4 / 6) * h, y0 + (ky1 / 6 + ky2 / 3 + ky3 / 3 + ky4 / 6) * h


def Simulate(x0, y0, h, N, f1, f2, method="symplectic4"):
    ret = []
    x, y = x0, y0
    ret.append(np.array([x, y]))
    for i in range(N):
        x, y = {
            "symplectic4": Symplectic4,
            "RK4": RK4,
        }.get(method, "RK4")(x, y, h, f1, f2)
        ret.append(np.array([x, y]))
    if (np.ndim(x) == 0):
        ret = np.array(ret).reshape(N + 1, 2)
    else:
        ret = np.array(ret).reshape(N + 1, 2 * len(x))
    return ret
