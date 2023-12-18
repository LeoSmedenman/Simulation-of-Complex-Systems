#Leonard Smedenman
"#################################################################################"
"________________________________Optical Tweezers_________________________________"
"#################################################################################"

import numpy as np
from matplotlib import pyplot as plt

### Coefficients of the simulation ###

T = 300
kB = 1.380649e-23      # Boltzmann's constant
R = 1e-6               # Radius of the Brownian particle
kx = 1e-6              # Stiffness of the optical trap along x
ky = 0.25e-6           # Stiffness of the optical trap along y
eta = 0.001
gamma = 6*np.pi*eta*R  # Drag coefficient of the medium
N = int(1e5)           # Simulation steps
dt = 1e-3              # Time step

x = np.zeros(N)    # Initiated trajectory array x
y = np.zeros(N)    # Initiated trajectory array y
Wx = np.random.randn(N)  # Gaussian distributed random numbers
Wy = np.random.randn(N)  # Gaussian distributed random numbers

for i in range(N-1):
    x[i+1] = x[i] - kx*x[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*Wx[i]      # Overdamped Langevin equation x
    y[i+1] = y[i] - ky*y[i]*dt/gamma + np.sqrt(2*kB*T*dt/gamma)*Wy[i]      # Overdamped Langevin equation y


plt.figure(figsize=(10,10))
plt.plot(x*1e9,y*1e9,'.',markersize=0.6)
plt.axis('equal')
plt.show()


#b)


def probability_distribution(trajectory, kB, T, k):
    bins = np.linspace(min(trajectory), max(trajectory), 100)
    prob_distribution, bins = np.histogram(trajectory, bins=bins, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    theoretical_distribution = np.exp(-0.5 * k * bin_centers ** 2 / (kB * T))
    theoretical_distribution /= np.sum(theoretical_distribution * np.diff(bins))
    return bin_centers, prob_distribution, theoretical_distribution


x_centers, px, px_theoretical = probability_distribution(x, kB, T, kx)
y_centers, py, py_theoretical = probability_distribution(y, kB, T, ky)

plt.figure(figsize=(12, 6))
plt.plot(x_centers, px, label='p(x) Simulation')
plt.plot(x_centers, px_theoretical, label='p(x) Theoretical', linestyle='dashed')
plt.plot(y_centers, py, label='p(y) Simulation')
plt.plot(y_centers, py_theoretical, label='p(y) Theoretical', linestyle='dashed')
plt.xlabel('Position')
plt.ylabel('Probability')
plt.title('Probability Distribution of the Particle in an Optical Trap')
plt.legend()
plt.show()


#c)
def positional_autocorrelation(trajectory, gamma, k, kB, T, dt):
    tau = gamma / k
    time_points = np.arange(0, len(trajectory) * dt, dt)
    C = kB * T / k * np.exp(-time_points / tau)
    return time_points, C

time_points, empirical_Cx = positional_autocorrelation(x, gamma, kx, kB, T, dt)
_, empirical_Cy = positional_autocorrelation(y, gamma, ky, kB, T, dt)

tau_x = gamma / kx
tau_y = gamma / ky
theoretical_Cx = kB * T / kx * np.exp(-time_points / tau_x)
theoretical_Cy = kB * T / ky * np.exp(-time_points / tau_y)

plt.figure(figsize=(12, 6))
plt.plot(time_points, empirical_Cx, label='Cx Simulation')
plt.plot(time_points, theoretical_Cx, label='Cx Theoretical', linestyle='dashed', color='black')
plt.plot(time_points, empirical_Cy, label='Cy Simulation')
plt.plot(time_points, theoretical_Cy, label='Cy Theoretical', linestyle='dashed', color='black')
plt.xlabel('Time [s]')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Functions in X and Y')
plt.xlim(0, 0.3)
plt.legend()
plt.show()


#d)
stiffness_values = np.linspace(1e-7, 1e-6, 10)
variances = []

for kx_current in stiffness_values:
    x_temp = np.zeros(N)
    Wx_temp = np.random.randn(N)
    for i in range(N - 1):
        x_temp[i + 1] = x_temp[i] - kx_current * x_temp[i] * dt / gamma + np.sqrt(2 * kB * T * dt / gamma) * Wx_temp[i]
    variances.append(np.var(x_temp))


plt.figure(figsize=(8, 6))
plt.plot(stiffness_values, variances, marker='o')
plt.xlabel('Stiffness (N/m)')
plt.ylabel('Variance (m^2)')
plt.title('Variance of Particle Position vs. Trapping Stiffness')
plt.grid(True)
plt.show()
