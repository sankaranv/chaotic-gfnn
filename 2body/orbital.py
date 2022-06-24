import numpy as np
from numpy import linalg


def RotX(theta):
    S, C = np.sin(theta), np.cos(theta)
    return np.array([[1, 0, 0], [0, C, -S], [0, S, C]])


def RotZ(theta):
    S, C = np.sin(theta), np.cos(theta)
    return np.array([[C, -S, 0], [S, C, 0], [0, 0, 1]])


def Orb2PosVel(orbs, mu):
    a, e, i, Omega, omega, E = orbs

    n = np.sqrt(mu / a / a / a)

    r = a * (1.0 - e * np.cos(E))
    V = np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(E / 2.0)) * 2.0

    R = np.matmul(np.matmul(RotZ(Omega), RotX(i)), RotZ(omega))

    P, Q = R[:, 0], R[:, 1]

    pos = np.matmul(R, np.array([r * np.cos(V), r * np.sin(V), 0.0]))
    vel = n * a * a / r * (Q * np.sqrt(1 - e * e) * np.cos(E) - P * np.sin(E))

    return [pos, vel]


def PosVel2Orb(pos, vel, mu):
    if (pos.shape[0] < 3):
        pos = np.append(pos, 0)
        vel = np.append(vel, 0)
    r, v = pos, vel

    # kI, kJ = np.array([1, 0, 0]), np.array([0, 1, 0])
    kK = np.array([0, 0, 1])

    h = np.cross(r, v)
    n = np.cross(kK, h)

    r_norm, v_norm, h_norm, n_norm = linalg.norm(r), linalg.norm(
        v), linalg.norm(h), linalg.norm(n)

    epsilon = v_norm * v_norm / 2.0 - mu / r_norm
    a = -mu / 2.0 / epsilon
    val = 1.0 + 2.0 * epsilon * h_norm * h_norm / mu / mu
    if (val < 0):
        val = 0
    e = np.sqrt(val)
    vec_e = r * (v_norm * v_norm / mu - 1.0 / r_norm) - v * (np.dot(r, v) / mu)

    val = h[2] / h_norm
    if np.abs(val) <= 1.0:
        i = np.arccos(val)
    else:
        if val > 0.0:
            i = 0
        else:
            i = np.pi

    val = n[0] / n_norm
    if np.abs(val) <= 1.0:
        Omega = np.arccos(val)
    else:
        if val > 0.0:
            Omega = 0
        else:
            Omega = np.pi

    val = np.dot(n, vec_e) / n_norm / e
    if np.abs(val) <= 1.0:
        omega = np.arccos(val)
    else:
        if val > 0.0:
            omega = 0
        else:
            omega = np.pi
    if vec_e[2] < 0.0:
        omega = 2 * np.pi - omega

    val = np.dot(vec_e, r) / e / r_norm
    if np.abs(val) <= 1.0:
        true_anomaly = np.arccos(val)
    else:
        if val > 0.0:
            true_anomaly = 0
        else:
            true_anomaly = np.pi
    if np.dot(r, v) < 0.0:
        true_anomaly = 2 * np.pi - true_anomaly

    return np.array([a, e, i, Omega, omega, true_anomaly])
