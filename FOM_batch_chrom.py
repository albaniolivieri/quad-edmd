import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#TODO: make a function out of this

# Parameters
eps = 0.4
L = 10.5  # cm
#L = 0.105  # m
Pe = 2000
#D = 0.026  # m
D = 2.6
kappa = [0.1, 0.1]  # s^-1
#H = [[2.688, 0.1], [3.728, 0.3]]  # dimensionless
H = [[3.728, 0.3], [2.688, 0.1]]
#K = [[0.0466, 0.0336], [3, 1]]  # L/g
K = [[46.6, 33.6], [3000, 1000]]  # ml/g
# K = [[0.0336, 1], [0.0446, 3]]  # L/g
# Q = 1.696e-5
Q = 0.1018  # cm^3/s
# Q = 0.3
c_f = [2.9e-3, 2.9e-3]  # g/L
t_inj = 1.35  # s
n = 400
T = 15.0  # s

# Derived parameters
A_c = np.pi * D**2 / 4  # cm^2
eps_c = (eps - 1) / eps
dx = L / (n - 1)  # cm
alpha = 1 / (Pe * dx**2)  # diffusion coefficient
beta = 1 / dx  # convection coefficient
factor = L / (Q / (eps * A_c))  # unit adjustment for kappa in q equation

# Time array
t = np.linspace(0, T, 2000)  # coarser for plotting, solver adapts internally

# Inlet condition (sigmoid profile scaled by c_f)
S = 5  # steepness parameter (arbitrary, adjust as needed)
def c_inlet(t):
    return 1 / (1 + np.exp(-S * (t - t_inj)))

# Inlet condition
# def c_inlet(t):
#     return 1 if t <= t_inj else 0

# Equilibrium concentration q_i^Eq
def q_eq(c1, c2, i):
    denom1 = 1 + K[0][0] * c_f[0] * c1 + K[1][0] * c_f[1] * c2
    denom2 = 1 + K[0][1] * c_f[0] * c1 + K[1][1] * c_f[1] * c2
    return (H[i][0] * c1 / denom1) + (H[i][1] * c1 / denom2)

# ODE system
def odes(y, t):
    # Unpack state: c1, q1, c2, q2 each of length n
    c1 = y[:n]
    q1 = y[n:2*n]
    c2 = y[2*n:3*n]
    q2 = y[3*n:]
    
    # Derivatives
    dc1_dt = np.zeros(n)
    dq1_dt = np.zeros(n)
    dc2_dt = np.zeros(n)
    dq2_dt = np.zeros(n)
    
    # q equations
    for j in range(n):
        q1_eq = q_eq(c1[j], c2[j], 0)
        q2_eq = q_eq(c2[j], c1[j], 1)
        dq1_dt[j] = factor * kappa[0] * (q1_eq - q1[j])
        dq2_dt[j] = factor * kappa[1] * (q2_eq - q2[j])
    
    # c equations
    # j = 0 (x = 0)
    c1_in = c_inlet(t)
    c2_in = c_inlet(t)
    dc1_dt[0] = (-(c1[1]-c1[0])/dx + (1/Pe)*(c1[1]-2*c1[0])/dx**2 + 
                (1/(Pe*dx*(1+dx*Pe)))*(c1[0]+Pe*dx*c1_in)+
                eps_c * factor * kappa[1] * (q_eq(c1[0], c2[0], 1) - q1[0])) 
    dc2_dt[0] = (-(c2[1]-c2[0])/dx + (1/Pe)*(c2[1]-2*c2[0])/dx**2 + 
                (1/(Pe*dx*(1+dx*Pe)))*(c2[0]+Pe*dx*c2_in)+
                eps_c * factor * kappa[1] * (q_eq(c2[0], c1[0], 1) - q2[0])) 
    # dc1_dt[0] = (-beta * Pe * (c1[0] - c1_in) + 
                #  alpha * (c1[2] - 2*c1[0]*(1 + Pe*dx) + 2*Pe*dx*c1_in + c1[0]) + 
                #  eps_c * factor * kappa[0] * (q_eq(c1[0], c2[0], 0) - q1[0]))
    # dc2_dt[0] = (-beta * Pe * (c2[0] - c2_in) + 
                #  alpha * (c2[2] - 2*c2[0]*(1 + Pe*dx) + 2*Pe*dx*c2_in + c2[0]) + 
                #  eps_c * factor * kappa[1] * (q_eq(c2[0], c1[0], 1) - q2[0]))
    
    # j = 1 to n-2 (interior)
    for j in range(1, n-1):
        dc1_dt[j] = (-beta * (c1[j] - c1[j-1]) + 
                     alpha * (c1[j+1] - 2*c1[j] + c1[j-1]) + 
                     eps_c * factor * kappa[0] * (q_eq(c1[j], c2[j], 0) - q1[j]))
        dc2_dt[j] = (-beta * (c2[j] - c2[j-1]) + 
                     alpha * (c2[j+1] - 2*c2[j] + c2[j-1]) + 
                     eps_c * factor * kappa[1] * (q_eq(c2[j], c1[j], 1) - q2[j]))
    
    # j = n-1 (x = L)
    j = n-1
    dc1_dt[j] = (-beta * (c1[j] - c1[j-1]) + 
                 alpha * (c1[j-1] - c1[j]) + 
                 eps_c * factor * kappa[0] * (q_eq(c1[j], c2[j], 0) - q1[j]))
    dc2_dt[j] = (-beta * (c2[j] - c2[j-1]) + 
                 alpha * (c2[j-1] - c2[j]) + 
                 eps_c * factor * kappa[1] * (q_eq(c2[j], c1[j], 1) - q2[j]))
    
    return np.concatenate([dc1_dt, dq1_dt, dc2_dt, dq2_dt])

# Initial conditions
y0 = np.zeros(4*n)  # c1, q1, c2, q2 all zero initially

# Solve ODE
sol = odeint(odes, y0, t, rtol=1e-10, atol=1e-10)

# Extract solutions
c1 = sol[:, :n]
q1 = sol[:, n:2*n]
c2 = sol[:, 2*n:3*n]
q2 = sol[:, 3*n:]

# Plot concentration at outlet (x = L)
plt.plot(t, c1[:, -1], label='c1 at x=L (Component 1)')
plt.plot(t, c2[:, -1], label='c2 at x=L (Component 2)')
plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.title('Outlet Concentrations vs Time')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Spatial profile at final time
x = np.linspace(0, L, n)
plt.plot(x, c1[-1, :], label='c1 at t=10s')
plt.plot(x, c2[-1, :], label='c2 at t=10s')
plt.xlabel('Position (cm)')
plt.ylabel('Concentration')
plt.title('Spatial Concentration Profile at t=10s')
plt.legend()
plt.grid(True)
plt.show()