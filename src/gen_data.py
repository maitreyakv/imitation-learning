"""
Copyright (C) 2020 Maitreya Venkataswamy - All Rights Reserved
"""

__author__ = "Maitreya Venkataswamy"

import numpy as np
from numba import jit
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd

@jit(nopython=True)
def f_func(t, x, u, m_c, m_p, l):
    s = np.sin(x[1])
    c = np.cos(x[1])
    lam = m_c + m_p * s**2
    psi = u[0] + m_p * s * (l * x[3]**2 + 9.81 * c)
    eta = -u[0] * c - m_p * l * x[3]**2 * c * s - (m_c + m_p) * 9.81 * s
    z_ddot = psi / lam
    theta_ddot = eta / (l * lam)
    return np.array([x[2], x[3], z_ddot, theta_ddot])

@jit(nopython=True)
def Phi_func(x, u, dt, m_c, m_p, l):
    F_x = np.zeros((x.size, x.size))

    s = np.sin(x[1])
    c = np.cos(x[1])

    F_x[0,0] = 0.0
    F_x[0,2] = 1.0
    F_x[0,1] = 0.0
    F_x[0,3] = 0.0

    F_x[2,0] = 0.0
    term = (m_p * c * (l * x[3]**2 + 9.81 * c) - 9.81 * m_p * s**2) / (m_c + m_p * s**2) - (2.0 * m_p * c * s * (u + m_p * s * (l * x[3]**2 + 9.81 * c))) / (m_c + m_p * s**2)**2
    F_x[2,1] = term[0]
    F_x[2,2] = 0.0
    F_x[2,3] = (2.0 * l * m_p * x[3] * s) / (m_c + m_p * s**2)

    F_x[1,0] = 0.0
    F_x[1,2] = 0.0
    F_x[1,1] = 0.0
    F_x[1,3] = 1.0

    F_x[3,0] = 0.0
    F_x[3,2] = 0.0
    term = (u * s - 9.81 * c * (m_c + m_p) - l * m_p * x[3]**2 * c**2 + l * m_p * x[3]**2 * s**2) / (l * (m_c + m_p * s**2)) + (2.0 * m_p * c * s * (l * m_p * c * s * x[3]**2 + u * c + 9.81 * s * (m_c + m_p))) / (l * (m_c + m_p * s**2)**2)
    F_x[3,1] = term[0]
    F_x[3,3] = -(2.0 * m_p * x[3] * c * s) / (m_c + m_p * s**2)

    return np.eye(x.size) + F_x * dt

@jit(nopython=True)
def beta_func(x, u, dt, m_c, m_p, l):
    F_u = np.zeros((x.size, 1))

    F_u[0] = 0.0
    F_u[2] = 1.0 / (m_c + m_p * np.sin(x[1])**2)
    F_u[1] = 0.0
    F_u[3] = -np.cos(x[1]) / (l * (m_c + m_p * np.sin(x[1])**2))

    return F_u * dt

class CartPoleDynamics():
    def __init__(self, m_c, m_p, l):
        self.m_c = m_c
        self.m_p = m_p
        self.l = l

    def f(self, t, x, u):
        return f_func(t, x, u, self.m_c, self.m_p, self.l)

    def Phi(self, x, u, dt):
        return Phi_func(x, u, dt, self.m_c, self.m_p, self.l)

    def beta(self, x, u, dt):
        return beta_func(x, u, dt, self.m_c, self.m_p, self.l)


@jit(nopython=True)
def anglediff(a1, a2):
    return np.pi - np.abs(np.abs(a1 - a2) - np.pi)

@jit(nopython=True)
def phi_func(x_f, x_star, Q):
    dx = x_f - x_star
    dx[1] = anglediff(x_f[1], x_star[1])
    return 0.5 * dx.T @ Q @ dx

@jit(nopython=True)
def phi_x_func(x_f, x_star, Q):
    dx = x_f - x_star
    dx[1] = anglediff(x_f[1], x_star[1])
    return Q @ dx

class QuadraticCost():
    def __init__(self, Q, R):
        self.Q = Q
        self.R = R

    def phi(self, x_f, x_star):
        return phi_func(x_f, x_star, self.Q)

    def phi_x(self, x_f, x_star):
        return phi_x_func(x_f, x_star, self.Q)

    def phi_xx(self, x_f, x_star):
        return self.Q

    def L(self, x, u, dt):
        return (0.5 * u.T @ self.R @ u) * dt

    def L_x(self, x, u, dt):
        return np.zeros(x.size)

    def L_u(self, x, u, dt):
        return (self.R @ u) * dt

    def L_xx(self, x, u, dt):
        return np.zeros((x.size, x.size))

    def L_uu(self, x, u, dt):
        return self.R * dt

    def L_xu(self, x, u, dt):
        return np.zeros((x.size, u.size))

    def L_ux(self, x, u, dt):
        return np.zeros((u.size, x.size))


def ddp_control(x0, x_star, tf, N, dyn, cost, u_max, num_iter, alpha):
    t = np.linspace(0.0, tf, N)
    dt = t[1] - t[0]

    J = np.zeros(num_iter)

    x = np.zeros((N, x0.size))
    x_new = np.zeros((N, x0.size))
    x[0,:] = x0
    x_new[0,:] = x0

    u = 0.1 * np.random.uniform(-u_max, u_max, (N, 1))
    u_new = np.zeros((N, 1))

    for k in range(N-1):
        x[k+1,:] = x[k,:] + dyn.f(t[k], x[k,:], u[k,:]) * dt
        x_new[k+1,:] = x_new[k,:] + dyn.f(t[k], x_new[k,:], u[k,:]) * dt

    V = np.zeros((N, 1))
    V_x = np.zeros((N, x0.size))
    V_xx = np.zeros((N, x0.size, x0.size))

    Q_x = np.zeros((N, x0.size))
    Q_u = np.zeros((N, 1))
    Q_xx = np.zeros((N, x0.size, x0.size))
    Q_uu = np.zeros((N, 1, 1))
    Q_xu = np.zeros((N, x0.size, 1))
    Q_ux = np.zeros((N, 1, x0.size))

    gain_ff = np.zeros((N, 1))
    gain_fb = np.zeros((N, 1, x0.size))

    for i in range(num_iter):
        J[i] = cost.phi(x[N-1,:], x_star)
        for k in range(N-1):
            J[i] = J[i] + cost.L(x[k,:], u[k,:], dt)

        V[N-1] = cost.phi(x[N-1,:], x_star)
        V_x[N-1,:] = cost.phi_x(x[N-1,:], x_star)
        V_xx[N-1,:,:] = cost.phi_xx(x[N-1,:], x_star)

        for k in range(N-2,-1,-1):
            Q_x[k,:] = cost.L_x(x[k,:], u[k,:], dt) + \
                     dyn.Phi(x[k,:], u[k,:], dt).T @ V_x[k+1,:]
            Q_u[k,:] = cost.L_u(x[k,:], u[k,:], dt) + \
                     dyn.beta(x[k,:], u[k,:], dt).T @ V_x[k+1,:]
            Q_xx[k,:,:] = cost.L_xx(x[k,:], u[k,:], dt) \
                      + dyn.Phi(x[k,:], u[k,:], dt).T @ V_xx[k+1,:,:] \
                      @ dyn.Phi(x[k,:], u[k,:], dt)
            Q_uu[k,:,:] = cost.L_uu(x[k,:], u[k,:], dt) \
                      + dyn.beta(x[k,:], u[k,:], dt).T @ V_xx[k+1,:,:] \
                      @ dyn.beta(x[k,:], u[k,:], dt)
            Q_xu[k,:,:] = cost.L_xu(x[k,:], u[k,:], dt) \
                      + dyn.Phi(x[k,:], u[k,:], dt).T @ V_xx[k+1,:,:] \
                      @ dyn.beta(x[k,:], u[k,:], dt)
            Q_ux[k,:,:] = cost.L_ux(x[k,:], u[k,:], dt) \
                      + dyn.beta(x[k,:], u[k,:], dt).T @ V_xx[k+1,:,:] \
                      @ dyn.Phi(x[k,:], u[k,:], dt)

            gain_ff[k,:] = -np.linalg.solve(Q_uu[k,:,:], Q_u[k,:])
            gain_fb[k,:,:] = -np.linalg.solve(Q_uu[k,:,:], Q_ux[k,:,:])

            V_x[k,:] = Q_x[k,:] - Q_xu[k,:,:] @ (np.linalg.solve(Q_uu[k,:,:], Q_u[k,:]))
            V_xx[k,:,:] = Q_xx[k,:,:] - Q_xu[k,:,:] @ (np.linalg.solve(Q_uu[k,:,:], Q_ux[k,:,:]))

        for k in range(N-1):
            u_new[k,:] = u[k,:] + alpha * (gain_ff[k,:] + gain_fb[k,:,:] @ (x_new[k,:] - x[k,:]))
            u_new[k,:] = np.clip(u_new[k,:], -u_max, u_max)
            x_new[k+1,:] = x_new[k,:] + dyn.f(t[k], x_new[k,:], u_new[k,:]) * dt

        u[:] = u_new[:]
        x[:] = x_new[:]

    return x, u, J


def simulate(x0, tf, dt, dyn):
    N = int(tf / dt)
    x = np.zeros((N,x0.size))
    u = np.zeros((N,1))
    x[0,:] = x0
    for i in range(N-1):
        t0 = i*dt
        t1 = t0 + dt
        u[i,:] = 0.
        sol = solve_ivp(dyn.f, (t0, t1), x[i,:], args=(u[i,:],))
        x[i+1,:] = sol.y[:,-1]
    return x, u


def main():
    m_c = 1.
    m_p = 0.01
    l = 0.25
    dyn = CartPoleDynamics(m_c, m_p, l)

    dt = 1. / 30.
    tf = 2.

    Q = 1e1 * np.eye(4)# ; Q[0,0] = 0.; Q[1,1] = 0
    R = 1e-1 * np.eye(1)
    x_star = np.array([0., np.pi, 0., 0.,])
    cost = QuadraticCost(Q, R)

    fig1 = plt.figure()
    ax1 = plt.gca(projection='3d')
    fig2 = plt.figure()
    ax2 = plt.gca()

    num_sim = 500
    with open("../data/data.csv", 'w') as fp:
        fp.write("m_c={},m_p={},l={}\n".format(m_c, m_p, l))
        fp.write("group,z,theta,z_dot,theta_dot,u\n")
        for grp in tqdm(range(num_sim)):
            x0 = np.array([np.random.uniform(-2., 2.),
                           np.random.uniform(-np.pi/16, np.pi/16),
                           np.random.uniform(-0.1, 0.1),
                           np.random.uniform(-np.pi/16, np.pi/16)
                          ])
            x, u, J = ddp_control(x0, x_star, tf, int(tf / dt), dyn, cost, 10., 300, 0.08)
            ax1.plot(x[:,0], np.linspace(0, tf, x.shape[0]), x[:,1])
            ax2.semilogy(J)
            for k in range(x.shape[0]):
                fp.write("{},{},{},{},{},{}\n".format(grp, *x[k,:], u[k,0]))

    plt.show()

if __name__ == "__main__":
    main()
