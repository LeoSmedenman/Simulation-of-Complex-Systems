import numpy as np
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt


def restart():
    global S
    I = np.argsort((x - l / 2) * 2 + (y - l / 2) * 2)
    S = np.zeros(n)
    S[I[1:initial_infected]] = 1


# Parameters of the simulation
n =1000  # Number of agents
initial_infected = 10  # Initial infected agents
N = 10000  # Simulation time
l = 100  # Lattice size



# Physical parameters of the system
x = np.floor(np.random.rand(n) * l)  # x coordinates
y = np.floor(np.random.rand(n) * l)  # y coordinates
S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: recovered
I = np.argsort((x - l / 2) * 2 + (y - l / 2) * 2)
S[I[1:initial_infected]] = 1  # Infect agents that are close to center

beta_values = np.linspace(0, 1, 5)
gamma_values = np.linspace(0.05, 0.1, 5)
D = 0.8


for gamma in gamma_values:
    for beta in beta_values:
        D_infinity = []
        mu_values = np.linspace(0.0, 0.5, 20)
        for mu in mu_values:
            casualties = []

            restart()

            for _ in range(N):

                B = beta
                G = gamma
                M = mu

                steps_x_or_y = np.random.rand(n)
                steps_x = steps_x_or_y < D / 2
                steps_y = (steps_x_or_y > D / 2) & (steps_x_or_y < D)
                nx = (x + np.sign(np.random.randn(n)) * steps_x) % l
                ny = (y + np.sign(np.random.randn(n)) * steps_y) % l

                for i in np.where((S == 1) & (np.random.rand(n) < B))[0]:  # loop over infecting agents
                    S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1  # Susceptiples together with infecting agent becomes infected


                S[(S == 1) & (np.random.rand(n) < M)] = 3  # deceased
                S[(S == 1) & (np.random.rand(n) < G)] = 2  # Recovered


                x = nx  # Update x
                y = ny  # Update y
                deceased = np.sum(S==3)
                casualties.append(deceased)

            if casualties:
                D_infinity.append(np.max(casualties))
            else:
                D_infinity.append(0)


        plt.plot(mu_values, D_infinity)
        plt.xlabel('mu')
        plt.ylabel("Final Deceased individuals")
        max_death_index = np.argmax(D_infinity)
        severe_mu = mu_values[max_death_index]
        plt.title(f'beta={beta:.2f}, gamma={gamma:.2f}')
        plt.show()
        print("Most severe mu:", severe_mu)