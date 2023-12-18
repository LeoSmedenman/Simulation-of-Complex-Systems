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
initial_infected = 100  # Initial infected agents
N = 10000  # Simulation time
l = 100  # Lattice size

# Physical parameters of the system
x = np.floor(np.random.rand(n) * l)  # x coordinates
y = np.floor(np.random.rand(n) * l)  # y coordinates
S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: recovered
I = np.argsort((x - l / 2) * 2 + (y - l / 2) * 2)
S[I[1:initial_infected]] = 1  # Infect agents that are close to center

nx = x
ny = y

num_of_runs = 0
susceptible_count = []
infected_count = []
recovered_count = []

beta = 0.55
gamma = 0.01
D = 0.8
alpha = 0.0015


for _ in range(N):
    B = beta
    G = gamma

    steps_x_or_y = np.random.rand(n)
    steps_x = steps_x_or_y < D / 2
    steps_y = (steps_x_or_y > D / 2) & (steps_x_or_y < D)
    nx = (x + np.sign(np.random.randn(n)) * steps_x) % l
    ny = (y + np.sign(np.random.randn(n)) * steps_y) % l



    for i in np.where((S == 1) & (np.random.rand(n) < B))[0]:  # loop over infecting agents
        S[(x == x[i]) & (y == y[i]) & (S == 0)] = 1  # Susceptiples together with infecting agent becomes infected

    S[(S == 1) & (np.random.rand(n) < G)] = 2  # Recovery
    S[(S == 2) & (np.random.rand(n) < alpha)] = 0  #re-susceptible

    x = nx
    y = ny

    susceptible_count.append(np.sum(S == 0))
    current_num_of_infected = np.sum(S == 1)
    infected_count.append(current_num_of_infected)
    recovered_count.append(np.sum(S == 2))


plt.plot(susceptible_count, label='Susceptible')
plt.plot(infected_count, label='Infected')
plt.plot(recovered_count, label='Recovered')
plt.xlabel('Time Steps')
plt.ylabel('Number of Agents')
plt.title('SIR Model Over Time')
plt.legend()
plt.show()