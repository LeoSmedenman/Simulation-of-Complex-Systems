#Leonard Smedenman
"#################################################################################"
"________________________________Random Walks_____________________________________"
"#################################################################################"


import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sci


cases = ['Flip Coin', 'Gaussian distributed steps', 'Asymmetric steps']
N = 1000                                                                         # Number of steps
n = 1000                                                                         # Number of copies
r1 = np.random.choice([-1, 1], size=(N, n))                                      # Random steps with a coin flip
r2 = np.random.randn(N, n)                                                       # Random steps with a Gaussian distribution
r3 = np.random.choice([-1, (1-np.sqrt(3))/2, (1+np.sqrt(3))/2], size=(N, n))
C = ['#0090B3', '#34B400', '#E66C00']


plt.figure(figsize=(20,6))
l = np.arange(40)/4 - 5
for subplot, sequences in zip([1,2,3], [r1,r2,r3]):
    plt.subplot(1,3,subplot)
    plt.hist(sequences.reshape(sequences.size,),l,color=C[subplot-1])
    plt.xlabel(r"$x$",fontsize=12)
    plt.ylabel(r"$p(x)$",fontsize=12)
    plt.title(cases[subplot-1], fontsize=12)
plt.show()

"###################################################################"

x = [np.cumsum(r1, axis=0), np.cumsum(r2,axis=0), np.cumsum(r3,axis=0)]          # Computing trajectories from random numbers

plt.figure(figsize=(20,6))

for subplot, trajectories in zip([1, 2, 3], x):
    plt.subplot(1, 3, subplot)
    plt.plot(trajectories[:, :100], linewidth=0.1, color=C[subplot-1])
    # plt.ylim([-140, 140])
    plt.xlabel(r"Time", fontsize=12)
    plt.ylabel('$x$', fontsize=12)
    plt.title(cases[subplot-1], fontsize=12)

plt.show()

"###################################################################"

plt.figure(figsize=(20,6))

l = np.arange(-15,15)*10
for subplot,trajectories in zip([1,2,3],x):
    # h = np.histogram()[0]
    plt.subplot(1,3,subplot)
    plt.hist(trajectories[999,:],l,color=C[subplot-1])
    plt.xlabel(r"$x_{end}$",fontsize=12)
    plt.ylabel('$p(x_{end})$',fontsize=12)
    plt.title(cases[subplot-1], fontsize=12)

plt.show()
# sci.savemat('Data_fig_randomwalks.mat',{'r1': r1, 'r2': r2, 'r3': r3})


"#################################################################################"
"________________________________Exercise 5.2_____________________________________"
"#################################################################################"


time_step_sizes = [0.01, 0.05, 0.1]
T = 5
num_of_trajectories = 50
# num_of_trajectories = 10000
msd_list = []

for time_step_i, delta_t in enumerate(time_step_sizes, 1):

    num_of_steps = int(T / delta_t)

    trajectories = np.zeros((num_of_trajectories, num_of_steps))

    for i in range(num_of_trajectories):
        white_noise = np.random.normal(0, np.sqrt(delta_t), size=num_of_steps)

        trajectories[i] = np.cumsum(white_noise)

    msd = np.mean(trajectories**2, axis=0)
    msd_list.append(msd)



    plt.subplot(1, 3, time_step_i)
    for d_t_size in range(num_of_trajectories):
        plt.plot(np.linspace(0, T, num_of_steps), trajectories[d_t_size], linewidth = 0.2, color = C[time_step_i-1])

    plt.grid(True)
    plt.title(f"Trajectories for Δt = {delta_t}")
    plt.xlabel(r"Time", fontsize=12)
    plt.ylabel('$x$', fontsize=12)
plt.show()


plt.figure(figsize=(12, 8))
for time_step_i, delta_t in enumerate(time_step_sizes, 1):
    plt.subplot(1, 3, time_step_i)
    times = np.linspace(0, T, int(T / delta_t))
    plt.plot(times, msd_list[time_step_i-1], label=f"Δt = {delta_t}", color=C[time_step_i-1])

    plt.grid(True)
    plt.xlabel('Time t')
    plt.ylabel('MSD')
    plt.title('MSD vs Time for Different Δt')
    plt.legend()
plt.show()


"#################################################################################"
"________________________________Exercise 5.3_____________________________________"
"#################################################################################"


def langevin_equation(total_time, delta_t, number_of_steps, m, gamma, kB, T):
    x_list = [0, 0]
    x_list_without_mass = [0, 0]
    white_noise = np.random.normal(0, np.sqrt(delta_t), size=number_of_steps + 2)

    for i in range(2, number_of_steps + 1):
        # with mass
        term_1 = ((2 + delta_t * (gamma/m))/(1 + delta_t * (gamma/m))) * x_list[i-1]
        term_2 = 1/(1 + delta_t * (gamma/m)) * x_list[i-2]
        term_3 = (np.sqrt(2 * kB * T * gamma)) / (m * (1 + delta_t * (gamma / m))) * delta_t ** (3 / 2) * white_noise[i]
        x = term_1 - term_2 + term_3
        x_list.append(x)

        # without mass
        x_without_mass = x_list_without_mass[i-1] + np.sqrt(2 * kB * T * delta_t / gamma) * white_noise[i]
        x_list_without_mass.append(x_without_mass)

    return np.array(x_list[1:]), np.array(x_list_without_mass[1:])


def compute_msd(trajectory):
    n = len(trajectory)
    msd = np.zeros(n)
    for i in range(n):
        msd[i] = np.mean((trajectory[i:] - trajectory[:n-i])**2)
    return msd


# Constants
R = 1e-9
m = 1.11e-14
eta = 0.001
gamma = 6 * np.pi * eta * R
T = 300
tau = m/gamma
delta_t = 0.05 * tau
kB = 1.380649e-23  # Boltzmann's constant
num_simulations = 10


# Simulation and plotting for part a
time_factors = [1, 100]


for total_time_factor in time_factors:
    total_time = total_time_factor * tau
    number_of_steps = int(total_time / delta_t)

    x_with_mass, x_without_mass = langevin_equation(total_time, delta_t, number_of_steps, m, gamma, kB, T)
    time_points = np.linspace(0, total_time, number_of_steps)

    plt.figure(figsize=(12, 6))
    plt.plot(time_points, x_with_mass, label='Inertial')
    plt.plot(time_points, x_without_mass, label='Non-inertial')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title(f'Brownian Motion Simulation for Total Time = {total_time_factor}τ')
    plt.legend()
    plt.show()

msd_results_with_mass = []
msd_results_without_mass = []

total_time = 100 * tau  # total time for the simulation
number_of_steps = int(total_time / delta_t)  # correct number of steps based on total time and delta_t

for _ in range(num_simulations):
    x_with_mass, x_without_mass = langevin_equation(total_time, delta_t, number_of_steps, m, gamma, kB, T)
    msd_results_with_mass.append(compute_msd(x_with_mass))
    msd_results_without_mass.append(compute_msd(x_without_mass))

average_msd_with_mass = np.mean(msd_results_with_mass, axis=0)
average_msd_without_mass = np.mean(msd_results_without_mass, axis=0)

time_points = np.linspace(0, total_time, number_of_steps)

plt.figure(figsize=(12, 6))
plt.loglog(time_points/tau, average_msd_with_mass, label='Inertial')
plt.loglog(time_points/tau, average_msd_without_mass, label='Non-inertial')
plt.xlabel('Time (in multiples of τ)')
plt.ylabel('MSD [m²]')
plt.title('MSD Comparison Over Different Timescales')
plt.legend()
plt.show()


#______________C________#


very_long_time = 1000 * tau  # A very long trajectory for time-averaged MSD
very_long_steps = int(very_long_time / delta_t)

x_with_mass_long, x_without_mass_long = langevin_equation(very_long_time, delta_t, very_long_steps, m, gamma, kB, T)
time_averaged_msd_with_mass = compute_msd(x_with_mass_long)
time_averaged_msd_without_mass = compute_msd(x_without_mass_long)

ensemble_msd_results_with_mass = []
ensemble_msd_results_without_mass = []

for _ in range(num_simulations):
    x_with_mass, x_without_mass = langevin_equation(very_long_time, delta_t, very_long_steps, m, gamma, kB, T)
    ensemble_msd_results_with_mass.append(compute_msd(x_with_mass))
    ensemble_msd_results_without_mass.append(compute_msd(x_without_mass))

ensemble_averaged_msd_with_mass = np.mean(ensemble_msd_results_with_mass, axis=0)
ensemble_averaged_msd_without_mass = np.mean(ensemble_msd_results_without_mass, axis=0)

time_points_long = np.linspace(0, very_long_time, very_long_steps)

plt.figure(figsize=(12, 6))
plt.loglog(time_points_long/tau, time_averaged_msd_with_mass, label='Time-averaged MSD with Mass')
plt.loglog(time_points_long/tau, time_averaged_msd_without_mass, label='Time-averaged MSD without Mass')
plt.loglog(time_points_long/tau, ensemble_averaged_msd_with_mass, label='Ensemble-averaged MSD with Mass', linestyle='--')
plt.loglog(time_points_long/tau, ensemble_averaged_msd_without_mass, label='Ensemble-averaged MSD without Mass', linestyle='--')
plt.xlabel('Time (in multiples of τ)')
plt.ylabel('MSD [m²]')
plt.title('Time-averaged vs Ensemble-averaged MSD Comparison')
plt.legend()
plt.show()