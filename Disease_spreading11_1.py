import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk

res = 500  # Animation resolution
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))
tk.configure(background='white')
canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res)
ccolor = ['#0008FF', '#DB0000', '#12F200']


def restart():
    global S
    I = np.argsort((x - l / 2) * 2 + (y - l / 2) * 2)
    S = np.zeros(n)
    S[I[1:initial_infected]] = 1


rest = Button(tk, text='Restart', command=restart)
rest.place(relx=0.05, rely=.85, relheight=0.12, relwidth=0.15)
Beta = Scale(tk, from_=0, to=1, orient=HORIZONTAL, label='Infection probability', font=("Helvetica", 8),
             resolution=0.01)
Beta.place(relx=.22, rely=.85, relheight=0.12, relwidth=0.23)
Beta.set(0.4)  # Parameter slider for infection rate


# Parameters of the simulation
n = 1000  # Number of agents
initial_infected = 10  # Initial infected agents
N = 1000  # Simulation time
l = 100  # Lattice size

# Physical parameters of the system
x = np.floor(np.random.rand(n) * l)  # x coordinates
y = np.floor(np.random.rand(n) * l)  # y coordinates
S = np.zeros(n)  # status array, 0: Susceptiple, 1: Infected, 2: recovered
I = np.argsort((x - l / 2) * 2 + (y - l / 2) * 2)
S[I[1:initial_infected]] = 1  # Infect agents that are close to center

nx = x  # udpated x
ny = y  # updated y

particles = []
R = .5  # agent plot radius
for j in range(n):  # Generate animated particles in Canvas
    particles.append(canvas.create_oval((x[j]) * res / l,
                                        (y[j]) * res / l,
                                        (x[j] + 2 * R) * res / l,
                                        (y[j] + 2 * R) * res / l,
                                        outline=ccolor[0], fill=ccolor[0]))


beta = Beta.get()
gamma = 0.01
D = 0.8
num_of_runs = 2
susceptible_count = []
infected_count = []
recovered_count = []
num_of_steps = 0

t = 0 #time


while t < N:
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


    for j in range(n):
        canvas.move(particles[j], (nx[j] - x[j]) * res / l, (ny[j] - y[j]) * res / l)  # Plot update - Positions
        canvas.itemconfig(particles[j], outline='#303030', fill=ccolor[int(S[j])])  # Plot update - Colors
        tk.update()
        tk.title('Infected:' + str(np.sum(S == 1)))

    x = nx  # Update x
    y = ny  # Update y

    susceptible_count.append(np.sum(S == 0))
    current_num_of_infected = np.sum(S == 1)
    infected_count.append(current_num_of_infected)
    recovered_count.append(np.sum(S == 2))

    num_of_steps +=1
    if current_num_of_infected == 0:
        break

Tk.mainloop(canvas)  # Release animation handle (close window to finish)

plt.figure()
plt.plot(susceptible_count, label='Susceptible')
plt.plot(infected_count, label='Infected')
plt.plot(recovered_count, label='Recovered')
plt.xlabel('Time Steps')
plt.ylabel('Number of Agents')
plt.title('SIR Model Over Time')
plt.legend()
plt.show()



