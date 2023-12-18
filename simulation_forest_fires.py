# Simulation of Forest Fires
# Leonard Smedenman

import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk as itk
import time
from scipy.ndimage import label
import matplotlib.pyplot as plt
from scipy.stats import linregress


def identify_burning_cluster(S, lightning_location, N):
    # assign cluster labels
    clusters, _ = label(S)

    # set periodic boundary conditions and assign periodic adjacent cells to same cluster
    for i in range(N):
        if S[i, 0] == 1 and S[i, -1] == 1:
            label_1, label_2 = clusters[i, 0], clusters[i, -1]
            clusters[clusters == label_2] = label_1

        if S[0, i] == 1 and S[-1, i] == 1:
            label_1, label_2 = clusters[0, i], clusters[-1, i]
            clusters[clusters == label_2] = label_1

    burnt_cluster = clusters[lightning_location[0], lightning_location[1]]

    return burnt_cluster, clusters


def calculate_cCDF(fire_sizes):
    fire_sizes = np.sort(fire_sizes)
    cCDF = 1 - np.arange(1, len(fire_sizes) + 1) / len(fire_sizes)
    return cCDF


def calculate_relative_fire_size(fire_sizes, N):
    fire_sizes = np.sort(fire_sizes)
    relative_fire_size = np.array(fire_sizes) / N**2
    return relative_fire_size


def plot_fire_size_histogram(fire_sizes):
    num_of_bins = int(np.sqrt(len(fire_sizes)))  # setting number of bins to square root of the number of fires

    plt.hist(fire_sizes, bins=num_of_bins, color='red')

    plt.title('Fire Size Histogram')
    plt.xlabel('Fire Size')
    plt.ylabel('Frequency')
    plt.show()


def plot_log_log_plot(relative_fire_size1, cCDF1, label1, relative_fire_size2=None, cCDF2=None, label2=None):
    plt.loglog(relative_fire_size1, cCDF1, 'bo', label=label1)

    if relative_fire_size2 is not None and cCDF2 is not None:
        plt.loglog(relative_fire_size2, cCDF2, 'ro', label=label2)

    plt.xlabel('n/N^2')
    plt.ylabel('C(n)')
    plt.title('Fire Size Distributions')
    plt.legend()
    plt.show()


def perform_linear_regression(x_vals, y_vals):
    log_x = np.log(x_vals)
    log_y = np.log(y_vals)
    slope, intercept, _, _, _ = linregress(log_x, log_y)
    alpha = 1-slope
    return slope, intercept, alpha


def plot_fitted_line(data_x, data_y, fit_x, fit_line, alpha):
    plt.loglog(data_x, data_y, 'b.', label='Simulation data')
    plt.plot(fit_x, np.exp(fit_line), 'r-', label=f'Fit: α = {alpha:.2f}')
    plt.xlabel('Relative fire size (n/N^2)')
    plt.ylabel('CCDF')
    plt.title('CCDF vs. Relative Fire Size on a log-log scale')
    plt.legend()
    plt.show()


res = 500  # Animation resolution
tk = Tk()
tk.geometry(str(int(res * 1.1)) + 'x' + str(int(res * 1.3)))
tk.configure(background='white')

canvas = Canvas(tk, bd=2)  # Generate animation window
tk.attributes('-topmost', 0)
canvas.place(x=res / 20, y=res / 20, height=res, width=res)
ccolor = ['#0008FF', '#DB0000', '#12F200']


p_growth = 0.01
f_lightning = 0.2
N = 128  # Lattice size
S = np.zeros((N, N))  # Status array, 0: No trees, 1: Trees 2: Burned  3: Expanding fire
forest_image = np.zeros((N, N, 3))  # Image array for the forest


"#################################################################################"
"________________________________Run Simulation___________________________________"
"#################################################################################"
fire_count = 0  # Number of fire events
fire_sizes = []
fire_sizes_rnd_forest = []
num_of_time_steps = 0


while num_of_time_steps < 10000:
# while fire_count < 5000:

    S[(np.random.rand(N, N) < p_growth) & (S == 0)] = 1  # Apply tree growth with the corresponding probability
    lightning_location = (np.random.rand(2) * N).astype(int)  # Randomly select a lightning location
    if (S[lightning_location[0], lightning_location[1]] == 1) and (
            np.random.rand() < f_lightning):  # If lightning falls on a tree
        fire_count += 1  # Fire event

        burnt_cluster, tree_clusters = identify_burning_cluster(S, lightning_location, N)
        current_fire_size = int(np.sum(S[tree_clusters == burnt_cluster]))
        fire_sizes.append(current_fire_size)

        number_of_trees = int(np.sum(S)) # for ex 3.5 store number of trees before trees start burning.

        random_forest = np.zeros((N, N))
        random_trees = np.random.choice(N*N, number_of_trees, replace=False)
        random_forest[np.unravel_index(random_trees, (N,N))] = 1

        random_lightning_location = (np.random.choice(N), np.random.choice(N))

        while random_forest[random_lightning_location] != 1:
            random_lightning_location = (np.random.choice(N), np.random.choice(N))

        burnt_cluster_rnd_forest, tree_clusters_rnd_forest = identify_burning_cluster(random_forest, random_lightning_location, N)
        current_fire_size_rnd_forest = int(np.sum(random_forest[tree_clusters_rnd_forest == burnt_cluster_rnd_forest]))
        fire_sizes_rnd_forest.append(current_fire_size_rnd_forest)

        S[tree_clusters == burnt_cluster] = 2 # set the affected trees afire

    forest_image[:, :, :] = 0  # Create image object for the forest, background black
    forest_image[:, :, 0] = (S == 2) * 255  # Burned trees are red
    forest_image[:, :, 1] = (S == 1) * 255  # Grown trees are green
    img = itk.PhotoImage(Image.fromarray(np.uint8(forest_image), 'RGB').resize((res, res)))
    canvas.create_image(0, 0, anchor=NW, image=img)
    tk.title('Fires:' + str(fire_count))
    tk.update()

    if sum(sum(S == 2)) > 0:
        # time.sleep(0.05)
        S[tree_clusters == burnt_cluster] = 0  # Burned trees will go back to status 0 (no trees)

    num_of_time_steps += 1

Tk.mainloop(canvas)  # Release animation handle (close window to finish)


plot_fire_size_histogram(fire_sizes)


empirical_cCDF = calculate_cCDF(fire_sizes)
empirical_n_N2 = calculate_relative_fire_size(fire_sizes, N)

random_cCDF = calculate_cCDF(fire_sizes_rnd_forest)
random_n_N2 = calculate_relative_fire_size(fire_sizes_rnd_forest, N)


plot_log_log_plot(empirical_n_N2, empirical_cCDF, 'C(n)')
plot_log_log_plot(empirical_n_N2, empirical_cCDF, 'Evolved forest', random_n_N2, random_cCDF, 'Random forest')

"#################################################################################"
"________________________Exercise 3.6 ____________________________________________"
"#################################################################################"

cutoff = np.max([np.argmax(empirical_n_N2 > 1/N), 1]) # cutoff to avoid finite size effects
fit_relative_sizes = empirical_n_N2[:cutoff]
fit_cCDF = empirical_cCDF[:cutoff]

slope, intercept, alpha = perform_linear_regression(fit_relative_sizes, fit_cCDF)

line_x = np.log(empirical_n_N2)  #Log of entire range of x-values
line_y = slope * line_x + intercept  #y-values using the fitted slope and intercept

plot_fitted_line(empirical_n_N2, empirical_cCDF, empirical_n_N2, line_y, alpha)


"#################################################################################"
"________________________Exercise 3.7 ____________________________________________"
"#################################################################################"
synthetic_alpha = 1.15
x_min = 1
synthetic_data_size = 100000
r = np.random.random(synthetic_data_size)
synthetic_data = x_min*(1-r)**(-1/(synthetic_alpha-1))
synthetic_data = synthetic_data[synthetic_data <= N**2] #extract possible sizes
synthetic_n_N2 = calculate_relative_fire_size(synthetic_data, N)
synthetic_cCDF = calculate_cCDF(synthetic_data)

line_y_for_synthetic = slope*np.log(synthetic_n_N2)+intercept
plot_fitted_line(synthetic_n_N2, synthetic_cCDF, synthetic_n_N2, line_y_for_synthetic, synthetic_alpha)



"#################################################################################"
"________________________Exercise 3.8 ____________________________________________"
"#################################################################################"

print(f"Exponent (α): {alpha} for N: {N}") #note the alpha values

lattice_sizes = np.array([16, 32, 64, 128, 256, 512, 1024])
alpha_mean = [1.2654471043977358, 1.2213742454707189, 1.1960686557199194, 1.1773247684497753, 1.1705400566470294, 1.1757406257692882, 1.1721045920103432]

plt.plot(1/lattice_sizes, alpha_mean, 'bo', label='alphaN vs N^-1')
slope, intercept,_,_,_ = linregress(1 / lattice_sizes, alpha_mean)

N_inv_extended = np.linspace(0, max(1 / lattice_sizes), 100)

alpha_extrapolated = slope * N_inv_extended + intercept

plt.plot(N_inv_extended, alpha_extrapolated, 'r-', label=f'Extrapolation (y={slope:.2f}x + {intercept:.2f})')

plt.xlabel('1/N')
plt.ylabel('Mean α_N')
plt.title('Extrapolation of α_N to N → ∞')
plt.legend()
plt.show()