import pandas as pd
import os
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import spacy
from spacy.training.example import Example
from spacy.util import minibatch
import random
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score
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
from scipy.stats import linregress
from matplotlib.ticker import MaxNLocator
from scipy.special import gamma
import math
from scipy.stats import linregress, t
from decimal import Decimal, getcontext
import warnings
import numpy as np
from scipy.stats import linregress
from decimal import Decimal, getcontext

# Suppress all warnings
warnings.filterwarnings("ignore")

# Read the Excel file
os.chdir("/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/webofscience")

# Load the three sheets into separate dataframes
df_air = pd.read_excel("Network.xlsx", sheet_name='Air')
df_ocean = pd.read_excel("Network.xlsx", sheet_name='Ocean')
df_land = pd.read_excel("Network.xlsx", sheet_name='Land')
df_social = pd.read_excel("Network.xlsx", sheet_name='Social')




# Testing if removing repeated V-E pairs has any influence
Clean = 0
if Clean == 1:
    df_air = df_air.drop_duplicates(subset=['V', 'E'])
    df_land = df_land.drop_duplicates(subset=['V', 'E'])
    df_ocean = df_ocean.drop_duplicates(subset=['V', 'E'])
    df_social = df_social.drop_duplicates(subset=['V', 'E'])










# Define a list of markers (you can add more symbols as needed)
markers = ["o", "s", "^", "D", "p", "*", "H", "v", "<", ">"]

# Create a function to assign a marker symbol based on the Type
def get_marker(type_val, markers):
    # Map the 'Type' to a unique marker
    unique_types = list(type_val.unique())  # Get unique types in the dataframe
    marker_map = {unique_types[i]: markers[i % len(markers)] for i in range(len(unique_types))}
    return marker_map


# Assuming the dataframes df_land, df_air, df_ocean, df_social are already loaded

# Create a figure with 2 rows and 2 columns for the grid layout
fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=600, layout="compressed")





# Subplot (a) - Efficiency of Dimensions (y = ln(x) / x)
x = np.linspace(1, 5, 400)  # Avoid 0 to handle log(0) issue
y = np.log(x) / x

# Plot the function
axes[0, 0].plot(x, y, label=r"$y = \frac{\ln(D)}{D}$", color='purple')
axes[0, 0].set_title(r"$\mathbf{A}$ Efficiency of dimensions", 
                     fontsize=18, loc='left')

axes[0, 0].set_xlabel(r"Dimension ($D$)", fontsize=18)
axes[0, 0].set_ylabel(r"Efficiency ($y$)", fontsize=18)
axes[0, 0].set_xlim(-0.5, 4.5)  # Set x-range from 0 to 4
axes[0, 0].set_ylim(0, 0.59)  # Auto adjust y-axis limits

# Add label for the function
axes[0, 0].legend(fontsize=22, loc="upper left")

# Make sure to format the axis ticks as integers or reasonable step sizes
axes[0, 0].tick_params(axis='both', labelsize=18)
axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=11))  # Adjust nbins as needed

# Draw a vertical dashed line at x = e
x_e = np.e
y_e = np.interp(x_e, x, y)  # Get the corresponding y-value from V_d at x = e
axes[0, 0].vlines(x_e, 0, y_e, colors='k', linestyle='--')  # Black dashed line
axes[0, 0].hlines(y_e, -10, 1000, colors='k', linestyle='--')  # Black dashed line

# Annotate at x = e (on the x-axis)
axes[0, 0].annotate("e", xy=(x_e, 0), xytext=(x_e, -0.038),
                    textcoords='data', fontsize=18,
                    ha='center', va='bottom', color='black')
axes[0, 0].annotate("Maximum\nefficiency", xy=(x_e, y_e), xytext=(x_e, 0.52),
                    textcoords='data', fontsize=18,
                    ha='center', va='bottom', color='black',
                    arrowprops=dict(facecolor='black', 
                                    edgecolor='black', 
                                    arrowstyle='->', lw=1.5))
axes[0, 0].annotate("Point\n(0$D$)", xy=(0, 0), xytext=(0, 0.25),
                    textcoords='data', fontsize=18,
                    ha='center', va='bottom', color='black')
axes[0, 0].annotate("Line\n(1$D$)", xy=(1, 0), xytext=(1, 0.28),
                    textcoords='data', fontsize=18,
                    ha='center', va='bottom', color='black')
axes[0, 0].annotate("Disk\n(2$D$)", xy=(2, 0), xytext=(2, 0.42),
                    textcoords='data', fontsize=18,
                    ha='center', va='bottom', color='black')
axes[0, 0].annotate("Ball\n(3$D$)", xy=(3, 0), xytext=(3, 0.43),
                    textcoords='data', fontsize=18,
                    ha='left', va='bottom', color='black')



axes[0, 0].annotate("•", xy=(0, 0), xytext=(0, 0.2),
                    textcoords='data', fontsize=28,
                    ha='center', va='bottom', color='black',
                    arrowprops=dict(facecolor='black', 
                                    edgecolor='black', 
                                    arrowstyle='->', lw=1.5))
axes[0, 0].annotate("────", xy=(1, 0), xytext=(1, 0.25),
                    textcoords='data', fontsize=18,
                    ha='center', va='bottom', color='black',
                    arrowprops=dict(facecolor='black', 
                                    edgecolor='black', 
                                    arrowstyle='->', lw=1.5))
axes[0, 0].annotate("○", xy=(2, 0), xytext=(2, 0.38),
                    textcoords='data', fontsize=28,
                    ha='center', va='bottom', color='black')
axes[0, 0].annotate("⨀", xy=(3, 0), xytext=(3.2, 0.38),
                    textcoords='data', fontsize=28,
                    ha='center', va='bottom', color='black')
axes[0, 0].annotate("Hyper-\nsphere $(nD)$", xy=(3.75, 0), xytext=(3.75, 0.2),
                    textcoords='data', fontsize=18,
                    ha='center', va='bottom', color='black')
axes[0, 0].annotate("⨂", xy=(3.75, 0), xytext=(3.75, 0.15),
                    textcoords='data', fontsize=28,
                    ha='center', va='bottom', color='black')


x_positions = [2, 3]

for i, x_pos in enumerate(x_positions):
    # Get the corresponding sqrt_V_d value at the given x position
    y_val = np.interp(x_pos, x, y)  # Interpolate to get the y value of sqrt_V_d at x_pos
    
    # Plot vertical dashed line from y=0 to the sqrt_V_d value
    axes[0, 0].vlines(x_pos, 0, y_val, colors='black', linestyle='--')
    
    # Plot scatter point
    axes[0, 0].scatter(x_pos, y_val, color='purple', zorder=5)











# Subplot (b) - Hypersphere Volume Factor and its square root
x = np.linspace(0.01, 11, 4000)  # Avoid 0 since Gamma(0) is undefined
d = 3  # Example dimension, you can change this to any dimension you like

# Hypersphere Volume Factor (V_d(x)) for dimension d
V_d = (math.pi**(x/2)) / gamma(x/2 + 1)

# Square root of the volume factor
ln_V_d = np.log(V_d)
axes[0, 1].set_xlim(0, 10)
axes[0, 1].set_ylim(0, 5.5)
# Plot both the volume factor and its square root
axes[0, 1].plot(x, V_d, 
                label=r"$V_D = \frac{\pi^{D/2}}{\Gamma(D/2+1)}$", 
                color="b")
axes[0, 1].plot(x, ln_V_d, 
                label=r"$\ln{V_D}$", color="r")

# Add labels and title
axes[0, 1].set_title(r"$\mathbf{B}$ Hypersphere volume factor", 
                     fontsize=18, loc='left')
axes[0, 1].set_xlabel(r"Dimension ($D$)", fontsize=18)
axes[0, 1].set_ylabel(r"Volume ($V$)", fontsize=18)


# Define the x positions where you want to place the vertical dashed lines
x_positions = [2, np.e, 3, 5, 9.5]

## Plot vertical dashed lines and scatter points at each of these positions
for i, x_pos in enumerate(x_positions):
    # Get the corresponding sqrt_V_d value at the given x position
    y_val = np.interp(x_pos, x, ln_V_d)  # Interpolate to get the y value of sqrt_V_d at x_pos
    
    # Plot vertical dashed line from y=0 to the sqrt_V_d value
    axes[0, 1].vlines(x_pos, 0, y_val, colors='black', linestyle='--')
    
    # Plot scatter point
    axes[0, 1].scatter(x_pos, y_val, color='red', zorder=5)

    # Annotate based on the index
    if i == 0:  # For the first point (x=2)
        axes[0, 1].annotate(f'', 
                            (x_pos, y_val), fontsize = 18,
                            textcoords="offset points", 
                            xytext=(0, 20), ha='center')
    elif i == 1:  # For the second point (x=e)
        axes[0, 1].annotate(f'', 
                            (x_pos, y_val), fontsize = 18,
                            textcoords="offset points", 
                            xytext=(0, 30), ha='center',
                            arrowprops=dict(facecolor='black', 
                                            edgecolor='black', 
                                            arrowstyle='->', lw=1.5))
    elif i == 2:  # For the third point (x=3)
        axes[0, 1].annotate(f'Avian\n{y_val:.3f}', 
                            (x_pos, y_val), fontsize = 18,
                            textcoords="offset points", 
                            xytext=(0, 80), ha='center',
                            arrowprops=dict(facecolor='black', 
                                            edgecolor='black', 
                                            arrowstyle='->', lw=1.5))
    elif i == 4:  # For the last point (x=9.5)
        axes[0, 1].annotate(f'', 
                            (x_pos, y_val), fontsize = 18,
                            textcoords="offset points", 
                            xytext=(0, 40), ha='center',
                            arrowprops=dict(facecolor='black', 
                                            edgecolor='black', 
                                            arrowstyle='->', lw=1.5))
    elif i == 3:
        axes[0, 1].annotate(f'Maximum volume\n{y_val:.3f}', 
                            (x_pos, y_val), fontsize = 18,
                            textcoords="offset points", 
                            xytext=(0, 110), ha='center',
                            arrowprops=dict(facecolor='black', 
                                            edgecolor='black', 
                                            arrowstyle='->', lw=1.5),)



# Add a legend
axes[0, 1].legend(loc='upper left', fontsize=22)

axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=11))  # Adjust nbins as needed

# Label e on the x-axis at x = e
e_val = np.e  # Euler's number (e)
axes[0, 1].annotate('e', 
                    xy=(e_val, 0.1),  # Position where you want to annotate (x=e, y=0)
                    xytext=(e_val, -0.22),  # Position where text will appear (to the right of e)
                    textcoords='data',  # Coordinates are in data units
                    fontsize=18,  # Set font size for the annotation
                    ha='center',  # Horizontal alignment of the text
                    va='center')  # Vertical alignment of the text
axes[0, 1].annotate('Social\n1.071', 
                    xy=(9, 2),   # Position where text will appear (to the right of e)
                    textcoords='data',  # Coordinates are in data units
                    fontsize=18,  # Set font size for the annotation
                    ha='center',  # Horizontal alignment of the text
                    va='center')  # Vertical alignment of the text
axes[0, 1].annotate('Ocean\n1.365', 
                    xy=(np.e-0.55, 2.15),   # Position where text will appear (to the right of e)
                    textcoords='data',  # Coordinates are in data units
                    fontsize=18,  # Set font size for the annotation
                    ha='center',  # Horizontal alignment of the text
                    va='center')  # Vertical alignment of the text
axes[0, 1].annotate('Land\n1.145', 
                    xy=(2-0.5, 1.5),   # Position where text will appear (to the right of e)
                    textcoords='data',  # Coordinates are in data units
                    fontsize=18,  # Set font size for the annotation
                    ha='center',  # Horizontal alignment of the text
                    va='center')  # Vertical alignment of the text







# Set the titles for each subplot with font size 14
axes[0, 2].set_title(f"$\mathbf{{C}}$ Land systems ($n$={len(df_land):,})", 
                     fontsize=18, loc='left')
axes[1, 1].set_title(f"$\mathbf{{E}}$ Avian systems ($n$={len(df_air):,})", 
                     fontsize=18, loc='left')
axes[1, 0].set_title(f"$\mathbf{{D}}$ Ocean systems ($n$={len(df_ocean):,})", 
                     fontsize=18, loc='left')
axes[1, 2].set_title(f"$\mathbf{{F}}$ Social systems ($n$={len(df_social):,})", 
                     fontsize=18, loc='left')
# Set the global font size for the plot
plt.rcParams.update({'font.size': 18})  # This will increase the font size for labels, titles, etc.

# Function to plot log10 of V vs E with linear regression and formula
def plot_log10(df, ax, color):
    # Filter out rows where 'V' or 'E' are 0 or NaN to avoid issues with log10
    df_filtered = df[(df['V'] > 0) & (df['E'] > 0)]
    
    # Calculate log10 of V and E
    log_V = np.log10(df_filtered['V'])
    log_E = np.log10(df_filtered['E'])
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(log_V, log_E)
    
    x_vals = np.linspace(0, 111, 100)
    # Plot the linear regression line
    ax.plot(x_vals, slope * x_vals + intercept, 
            color=color, linestyle='--')
    
    # Plot the reference line y = x
    ax.plot([0, 111], [0, 111], 
            color='black', linestyle='-')
    
    # Set the same limits for both axes based on data
    min_val = min(min(log_V), min(log_E))
    max_val = max(max(log_V), max(log_E))
    ax.set_xlim(0, max_val+1)
    ax.set_ylim(0, max_val+1)
    
    # Set the labels and title
    ax.set_xlabel(r'$\log_{10}$(Nodes)', fontsize = 18)
    ax.set_ylabel(r'$\log_{10}$(Edges)', fontsize = 18)
    
    # Remove grid and frame
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    
    # Add the formula label (without box)
    ax.legend(loc='upper left')
    
    
    # Create regression line
    
    y_vals = slope * x_vals + intercept
    
    # Calculate the standard error for prediction
    y_pred = slope * log_V + intercept
    residuals = log_E - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Compute confidence interval (95% CI)
    ci_lower = y_vals - 1.96 * std_residual
    ci_upper = y_vals + 1.96 * std_residual
    
    # Fill the 95% confidence interval shading
    ax.fill_between(x_vals, ci_lower, ci_upper, color=color, alpha=0.2)
    
    
    # Get marker map for the types in the dataframe
    marker_map = get_marker(df['Type'], markers)

    # Plot the scatter points with different markers for each 'Type'
    for type_val, marker in marker_map.items():
        type_df = df[df['Type'] == type_val]
        ax.scatter(np.log10(type_df['V']), np.log10(type_df['E']), label=type_val, marker=marker, alpha=0.7)

    # Add legend in the lower-right corner
    ax.legend(loc='lower right', fontsize=16)
    
    
    # Degrees of freedom
    df = len(log_V) - 2
    # Calculate the t-value for 95% CI (two-tailed)
    t_value = t.ppf(1 - 0.025, df)  # 95% confidence level -> alpha = 0.05
    # Calculate the 95% confidence interval for the slope
    slope_ci = t_value * std_err
    slope_lower = slope - slope_ci
    slope_upper = slope + slope_ci
    
    p_value_decimal = Decimal(p_value)
    # Check if p-value is extremely small
    if p_value < 1e-300 and Clean == 0:
        p_value_str = "1e-300"  # or any other placeholder you prefer
    elif p_value < 1e-300 and Clean == 1:
        p_value_str = "1e-300"
    else:
        p_value_str = f"{p_value:.1e}"  # Format p-value in scientific notation


    
    if p_value>1e-300:
        ax.annotate(
        f'Slope = {slope:.3f} ± {slope_ci:.3f} (95% CI)\n$p_{{Slope}}$={p_value_str}\nR² = {r_value**2:.3f}',
        xy=(0, 1),  # Coordinates for upper-left corner
        xytext=(0.02, 0.97),  # Adjust this value to move the annotation
        textcoords='axes fraction',  # Relative positioning within the axes
        ha='left',  # Horizontal alignment
        va='top',  # Vertical alignment
        fontsize=18)  # Optional: add a box around the text
    else:
        ax.annotate(
        f'Slope = {slope:.3f} ± {slope_ci:.3f} (95% CI)\n$p_{{Slope}}$<1e-300\nR² = {r_value**2:.3f}',
        xy=(0, 1),  # Coordinates for upper-left corner
        xytext=(0.02, 0.97),  # Adjust this value to move the annotation
        textcoords='axes fraction',  # Relative positioning within the axes
        ha='left',  # Horizontal alignment
        va='top',  # Vertical alignment
        fontsize=18)  # Optional: add a box around the text
    
    print(p_value_decimal,"\n")


# Plot the data for each dataframe with its respective color
plot_log10(df_land, axes[0, 2], color='green')  # Land
plot_log10(df_air, axes[1, 1], color='#885749')   # Air
plot_log10(df_ocean, axes[1, 0], color='#37aefe')  # Ocean
plot_log10(df_social, axes[1, 2], color='orange')  # Social

# Adjust spacing between plots to reduce horizontal space
plt.subplots_adjust(wspace=0.05)  # Adjust hspace and wspace values as needed

# Save the plot as a PDF
if Clean ==1:
    pdf_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Fig3'.pdf"
else:
    pdf_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Fig3.pdf"

plt.savefig(pdf_file, format="pdf", bbox_inches="tight")  # Save as PDF with tight layout



axes[0, 1].tick_params(axis='both', labelsize=18)
axes[0, 2].tick_params(axis='both', labelsize=18)
axes[1, 0].tick_params(axis='both', labelsize=18)
axes[1, 1].tick_params(axis='both', labelsize=18)
axes[1, 2].tick_params(axis='both', labelsize=18)




# Show the plot
plt.tight_layout()
# Show the plot
plt.show()




print(len(df_land)+len(df_air)+len(df_ocean)+len(df_social))










