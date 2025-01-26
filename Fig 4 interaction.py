import spacy
import pandas as pd
from spacy.training import Example
from spacy.util import minibatch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
os.chdir("/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/webofscience")
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import xgi
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import spacy
import pandas as pd
from spacy.training import Example
from spacy.util import minibatch
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib as mpl
from decimal import Decimal, getcontext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


mpl.rcParams['agg.path.chunksize'] = 1000000000



fig = plt.figure(figsize=(25, 15), dpi = 600)
grid = fig.add_gridspec(2, 3,
                        height_ratios=[1, 1], width_ratios=[1, 1, 1])
# Adjust layout: Increase vertical space between rows

plt.style.use('default')  # Ensure default style for a white background
# Main plot (first row, spans all columns)
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)  # You can adjust this value (e.g., 0.5) to get more space


ax_1 = fig.add_subplot(grid[0, 0], projection='3d')

def menger_sponge_coordinates(level, x=0, y=0, z=0, size=3):
    if level == 0:
        return [(x, y, z, size)]
    
    new_size = size / 3
    sponge = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                # Skip the center cubes
                if (abs(dx) + abs(dy) + abs(dz)) > 1:
                    sponge.extend(
                        menger_sponge_coordinates(
                            level - 1, 
                            x + dx * new_size, 
                            y + dy * new_size, 
                            z + dz * new_size, 
                            new_size
                        )
                    )
    return sponge

def plot_menger_sponge(level):
    sponge = menger_sponge_coordinates(level)
    
    for x, y, z, size in sponge:
        # Plot a cube at each coordinate
        r = [x - size / 2, x + size / 2]
        ax_1.bar3d(x - size / 2, y - size / 2, z - size / 2, 
                 size, size, size, color = "#7aa9c4", shade=True)
    # Customize tick labels with fewer ticks and larger font size
    ax_1.set_xticks([-1, 0, 1])
    ax_1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_1.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax_1.tick_params(axis='both', which='major', labelsize=18)
    ax_1.set_box_aspect([1, 1, 1])  # Maintain aspect ratio
    

# Visualize the Menger sponge at level 2
plot_menger_sponge(3)


ax_1.annotate(f"$D=\log_{3}20≈$e\n(Menger sponge)", 
            xy=(0, 0.0), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=20, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

















ax_2 = fig.add_subplot(grid[0, 1])




n = 10
is_connected = False
while not is_connected:
    H_random = xgi.random_hypergraph(n, [0.3, 0.05, 0.01])
    is_connected = xgi.is_connected(H_random)
pos = xgi.barycenter_spring_layout(H_random, seed=11)


xgi.draw(
    H_random,
    pos=pos,
    ax=ax_2,
    node_size=H_random.nodes.degree,
    node_fc=H_random.nodes.clique_eigenvector_centrality,
)



ax_2.annotate(f"e<$D$<10\n(interdimensional motif)", 
            xy=(0.5, 0.0), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=20, 
            ha='center', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))







ax_3 = fig.add_subplot(grid[0, 2], projection='3d')
def generate_hypercube_vertices(dimension):
    """
    Generate vertices for a hypercube of a given dimension.
    """
    return np.array(list(product([0, 1], repeat=dimension)))

def project_to_3d(vertices, seed=42):
    """
    Project high-dimensional vertices onto 3D space using a random projection
    with a fixed random seed.
    """
    np.random.seed(14)  # Fix the random seed for consistent results
    projection_matrix = np.random.rand(3, vertices.shape[1]) - 0.5
    return vertices @ projection_matrix.T

def plot_10d_hypercube():
    """
    Plot a 10D hypercube projected into 3D space with smaller ranges.
    """
    dimension = 10
    
    # Generate 10D hypercube vertices
    vertices = generate_hypercube_vertices(dimension)
    
    # Project vertices into 3D space
    vertices_3d = project_to_3d(vertices, seed=42)
    
    # Plot the projected hypercube
    ax_3.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2], c="#7aa9c4", s=10)
    
    # Draw edges between vertices that differ by one bit
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            if np.sum(np.abs(v1 - v2)) == 1:  # Hamming distance of 1
                ax_3.plot(
                    [vertices_3d[i, 0], vertices_3d[j, 0]],
                    [vertices_3d[i, 1], vertices_3d[j, 1]],
                    [vertices_3d[i, 2], vertices_3d[j, 2]],
                    color='gray',
                    linewidth=0.5
                )
    
    # Adjust axis ranges for smaller view
    ax_3.set_xlim(-1, 1.2)
    ax_3.set_ylim(-1, 1)
    ax_3.set_zlim(-1, 1)
    # Customize tick labels with larger font size
    
    # Customize tick labels with fewer ticks and larger font size
    ax_3.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_3.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_3.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax_3.tick_params(axis='both', which='major', labelsize=18)
    
    ax_3.set_title("10D Hypercube Projected into 3D Space (Fixed Projection)")
    ax_3.set_box_aspect([1, 1, 1])  # Maintain aspect ratio
    

# Plot the 10D hypercube
plot_10d_hypercube()


ax_3.annotate(f"$D$=10\n(projected on 3D)", 
            xy=(0, 0.0), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=20, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))












ax_4 = fig.add_subplot(grid[1, 0])
# Read back the data from the CSV
plot_data = pd.read_csv("Spatiotemporal.csv")
x = plot_data["Temporal"]
y = plot_data["Spatial"]



















# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)
p_value_decimal = Decimal(p_value)
# Generate the regression line
x_new = np.linspace(-1, 1, 1000)
regression_line = intercept + slope * x_new


plt.style.use('default')  # Ensure default style for a white background
# Main plot (first row, spans all columns)
plt.scatter(x, 
            y,
            c = plot_data["z"], alpha = .2,
            s=.5,cmap = 'Spectral',marker = 'o')
plt.plot(x_new, regression_line, color="black")



# Set the x and y limits
plt.xlim(0, 0.28)  # Set x-axis limits from 2 to 8
plt.ylim(0, 0.4)  # Set y-axis limits from -0.5 to 1
# Define and apply logarithmic normalization to color scale
norm = mcolors.LogNorm(vmin=1, vmax=1000)  # Adjust `vmin` and `vmax` as needed

# Adding a color bar as a legend for the density
sm = plt.cm.ScalarMappable(cmap="Spectral", norm=norm)
sm.set_array([])  # Empty array for matplotlib compatibility with ScalarMappable
cbar = plt.colorbar(sm, ax=ax_4, orientation='vertical', pad=0.02)
cbar.set_label('Log density', fontsize = 18)
# Change the tick label size of the colorbar
cbar.ax.tick_params(labelsize=16)  # Change 12 to any desired font size

# Set axis labels with one decimal place and adjust the range
ax_4.set_xlabel('Temporal scale', 
                   fontsize=20)
ax_4.set_ylabel('Spatial scale',
                   fontsize=20)
# Customize the tick label size (x and y axes)
ax_4.tick_params(axis='both', labelsize=20)  # You can change 12 to any desired size
yticks = ax_4.get_yticks()
yticks = [tick for tick in yticks if tick != 0]  # Remove 0 from y-axis if present
# Add annotation for the formula and R^2 on the plot
Dim = np.exp(slope)
annotation_text = f"All systems\ny = {slope:.3f}x\n$p$<1e-300\n$R^2$={r_value**2:.4f}\n$n={len(y):,}$\n$D$={Dim:.1f}"

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
ax_4.annotate(annotation_text, 
            xy=(0.05, 0.98), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=20, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
ax_4.annotate("Slope: marginal increment\nof information with respect\nto measurement", 
            xy=(0.3, 0.2), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=20, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))


# Get the first color from the 'Set1' colormap
cmap = plt.cm.Set1
first_color = cmap(0)  # 0 corresponds to the first color in the colormap
print(p_value)









ax_5 = fig.add_subplot(grid[1, 1], projection="3d")
H = xgi.Hypergraph()
H.add_edges_from(
    [[1, 2, 3], [3, 4, 5], [3, 6], [6, 7, 8, 9], [1, 4, 10, 11, 12], [1, 4]]
)


xgi.draw_multilayer(H, ax=ax_5)




ax_5.text2D(0.5, 0.9, f"$A$ $priori$ information $\ln$$n$", 
            transform=ax_5.transAxes, ha='center', fontsize=20)
ax_5.text2D(0.5, 0.1, f"$A$ $posteriori$ information $\ln$$(n+m)$", 
            transform=ax_5.transAxes, ha='center', fontsize=20)
ax_5.text2D(0.5, 0, r"Increment of information $\ln$$({1+\frac{m}{n}})$", 
            transform=ax_5.transAxes, ha='center', fontsize=20)



ax_6 = fig.add_subplot(grid[1, 2])
# Get the first color from the 'Set1' colormap
cmap = plt.cm.Set1
first_color = cmap(0)  # 0 corresponds to the first color in the colormap





# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="upper left")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Aquaculture'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color=cmap(0), s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "upper left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(type_to_plot, 
            xy=(0.15, 0.15), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)





# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="center left")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Marine protected areas'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color=cmap(1), s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "upper left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate("MPA", 
            xy=(0.65, 0.15), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)









# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="lower center")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Desalination'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color=cmap(2), s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "upper left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(type_to_plot, 
            xy=(0.1, 0.15), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)





# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="lower right")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Ship operations'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color=cmap(6), s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "upper left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate("Ship\noperations", 
            xy=(0.95, 0.33), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='right', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)











# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="lower left")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Wind energy'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color=cmap(8), s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "lower left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate("Wind\nenergy", 
            xy=(0.95, 0.33), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='right', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)











# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="center right")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Land reclamation'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color=cmap(3), s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "lower left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate("Land\nreclamation", 
            xy=(0.95, 0.33), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='right', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)












# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="upper right")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Disaster recovery'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color=cmap(4), s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "lower left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate("Disaster\nrecovery", 
            xy=(0.95, 0.33), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='right', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)










# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="upper center")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Marine mining'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color=cmap(7), s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "lower left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate("Marine\nmining", 
            xy=(0.95, 0.33), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='right', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)
















# Add inset axes (smaller plot inside the main plot)
# We choose the position and size of the inset axes (left, bottom, width, height)
inset_ax = inset_axes(ax_6, width="31.5%", height="31.5%", loc="center")  # Change size as needed

# Filter the data by 'Type' (assuming Type is a column in your DataFrame)
type_to_plot = 'Freight shipping'  # Change this to the desired 'Type'
subset = plot_data[plot_data['Type'] == type_to_plot]

# Extract x and y values for the subset
x_inset = subset['Temporal']
y_inset = subset['Spatial']

# Plot the data for the subset inside the inset
inset_ax.scatter(x_inset, y_inset, color="#e0c470", s = 4)

# Fit the exponential curve for the subset
slope, intercept, r_value, p_value, std_err = linregress(x_inset, 
                                                         y_inset)
regression_line = intercept + slope * x_new
inset_ax.plot(x_new, regression_line, color="black", label=f"y = {slope:.3f}x\n$R^2 = {r_value**2:.4f}$")



inset_ax.legend(fontsize=18, frameon=False,
                loc = "lower left",
                handles=[])  # No legend box for inset

# Remove axis ticks and labels from the inset plot
inset_ax.set_xticks([])  # Remove x-axis ticks
inset_ax.set_yticks([])  # Remove y-axis ticks
inset_ax.set_xlabel('')  # Remove x-axis label
inset_ax.set_ylabel('')  # Remove y-axis label


# Add annotation for the formula and R^2 on the plot
annotation_text = f"y = {slope:.3f}x\n$R^2 = {r_value**2:.2f}$"
Dim = np.exp(slope)
# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate(f"$D$={Dim:.1f}", 
            xy=(0.6, 0.5), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

# Place the annotation at a suitable position, for example at (0.05, 0.95) in axis fraction
inset_ax.annotate("Freight\nshipping", 
            xy=(0.95, 0.33), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='right', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))

inset_ax.annotate(annotation_text, 
            xy=(0.02, 0.96), 
            xycoords='axes fraction',  # Position in axes fraction (0 to 1)
            fontsize=18, 
            ha='left', 
            va='top',
            bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.5'))
# Set the x and y limits
inset_ax.set_xlim(0, 0.38)  # Set x-axis limits from 2 to 8
inset_ax.set_ylim(0.01, 0.5)  # Set y-axis limits from -0.5 to 1
print(p_value<1e-300)



# Selecting the axis-X making the bottom and top axes False. 
ax_6.tick_params(axis='x', which='both', bottom=False, 
                top=False, labelbottom=False) 
  
# Selecting the axis-Y making the right and left axes False 
ax_6.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
ax_6.set_frame_on(False)





ax_1.set_title(r'$\mathbf{{A}}$ Approximation of ocean dimension', 
               fontsize=20, loc="center")
ax_2.set_title(r'$\mathbf{{B}}$ Interdimensional hypergraph', 
               fontsize=20, loc="center")
ax_3.set_title(r'$\mathbf{{C}}$ Approximation of social dimension', 
               fontsize=20, loc="center")
ax_4.set_title(r'$\mathbf{{D}}$ Informational dimension of all SOSs', 
               fontsize=20, loc="center")
ax_5.set_title(r'$\mathbf{{E}}$ Measurement of the dimension', 
               fontsize=20, loc="center")
ax_6.set_title(r'$\mathbf{{F}}$ Informational dimensions of various SOSs', 
               fontsize=20, loc="center")




plt.subplots_adjust(hspace=0.3, wspace=-0.05)
ax_6.set_xlabel(f'(All $p$-values < 1e-300)', 
                   fontsize=20)

# Save the plot as a PDF
jpeg_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Fig4.jpeg"
plt.savefig(jpeg_file, format="jpeg", bbox_inches="tight")  # Save as PDF with tight layout

# Show the plot
plt.tight_layout()
plt.show()








