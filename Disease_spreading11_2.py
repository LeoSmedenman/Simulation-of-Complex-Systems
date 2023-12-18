import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk


def restart():
    global S
    I = np.argsort((x - l / 2) * 2 + (y - l / 2) * 2)
    S = np.zeros(n)
    S[I[1:initial_infected]] = 1


# Parameters of the simulation
n = 1000  # Number of agents
initial_infected = 10  # Initial infected agents
N = 10  # Simulation time
l = 100  # Lattice size

# Physical parameters of the system
x = np.floor(np.random.rand(n) * l)  # x coordinates
y = np.floor(np.random.rand(n) * l)  # y coordinates
S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: recovered
I = np.argsort((x - l / 2) * 2 + (y - l / 2) * 2)
S[I[1:initial_infected]] = 1  # Infect agents that are close to center



beta_values = np.linspace(0, 1, 35)
gamma_values = np.linspace(0.01, 0.02, 35)
D = 0.8

num_of_runs = 3
num_of_steps = 0

t = 0 #time
beta_gamma_combined = []

R_infinity_mat = np.zeros((len(beta_values), len(gamma_values)))
for gamma_idx, gamma in enumerate(gamma_values):
    G = gamma
    averaged_R_infinity = []
    beta_gamma = beta_values / G

    beta_gamma_combined.extend(beta_gamma)

    for beta_idx, beta in enumerate(beta_values):
        B = beta
        R_infinity_runs = []
        for run in range(num_of_runs):
            restart()
            t=0
            recovered_count = []
            while t < N:
                steps_x_or_y = np.random.rand(n)
                steps_x = steps_x_or_y < D / 2
                steps_y = (steps_x_or_y > D / 2) & (steps_x_or_y < D)
                nx = (x + np.sign(np.random.randn(n)) * steps_x) % l
                ny = (y + np.sign(np.random.randn(n)) * steps_y) % l

                for i in np.where((S == 1) & (np.random.rand(n) < B))[0]:  # loop over infecting agents
                    S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1  # Susceptiples together with infecting agent becomes infected

                S[(S == 1) & (np.random.rand(n) < G)] = 2  # Recovery

                t += 1
                x = nx  # Update x
                y = ny  # Update y

            current_num_of_infected = np.sum(S == 1)
            recovered_count.append(np.sum(S == 2))



        R_infinity_runs.append(max(recovered_count))
        averaged_R_infinity.append(np.mean(R_infinity_runs))
        R_infinity_mat[beta_idx, gamma_idx] = np.mean(recovered_count)

    #11.2a,b
    #plt.scatter(beta_values, averaged_R_infinity, label=f"Gamma = {gamma:.2f}")
    plt.scatter(beta_gamma, averaged_R_infinity, label=f"Gamma = {gamma:.2f}")
plt.xlabel("beta/gamma")
plt.ylabel("R_{infinity}")
plt.xlabel('Infection Rate (Beta)')
plt.ylabel(' R_{infinity}')
plt.title('Final Number of Recovered Agents vs Infection Rate')
plt.legend()
plt.show()

print(R_infinity_mat)

# Tk.mainloop(canvas)  # Release animation handle (close window to finish)



# 11.2c
plt.imshow(R_infinity_mat, aspect='auto', extent=(beta_values[0], beta_values[-1], beta_values[0]/gamma_values[0], beta_values[-1]/gamma_values[-1]),
           origin='lower', cmap='hot')
plt.colorbar(label=r'R_{infinity}')
plt.xlabel('beta')
plt.ylabel('beta/gamma')
plt.title('Phase Diagram of R_{infinity} as a function of beta and gamma')
plt.show()
