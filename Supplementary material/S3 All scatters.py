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
import matplotlib.cm as cm
import scipy.stats as stats
from decimal import Decimal, getcontext
import scipy.stats as stats

main_directory = '/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/webofscience'

# Read back the data from the CSV
plot_data = pd.read_csv("Plot.csv")


fig = plt.figure(figsize=(12, 20), dpi = 300)
grid = fig.add_gridspec(8, 5)
# Adjust layout: Increase vertical space between rows
plt.subplots_adjust(hspace=0.25)  # You can adjust this value (e.g., 0.5) to get more space
folder_means = plot_data.groupby('Type',
                            as_index=False)[['SimOcean', 'SimSocial']].mean()

# Generate a color palette
palette = sns.color_palette("tab20", 78)  # "tab20" palette with 40 distinct colors

slope_list = []
intercept_list = []

for i in range(0,40):
    Type = folder_means['Type'][i]
    ax_scatter1 = fig.add_subplot(grid[i // 5, 
                                       i % 5])
    ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                        plot_data[plot_data["Type"] == Type]["SimOcean"], 
                        s = .5, color = palette[i])
    ax_scatter1.set_xlabel(Type)
    ax_scatter1.set_xticks([])
    ax_scatter1.set_yticks([])
    # Fit a line (linear fit in this case)
    p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                   plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
    y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
    # Perform linear regression using scipy to get p-value of the slope
    slope_type, intercept_type, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                        plot_data[plot_data["Type"] == Type]["SimOcean"])

    
    r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
    # Plot the fitted line
    ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
    N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])

    # Display the fitted formula in the bottom right corner
    formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, N={N:,}"
    ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    if p_value < 1e-300: 
        p_value_str = "1e-300"
    else:
        p_value_str = f"{p_value:.1e}"  # Format p-value in scientific notation

    
    
    if p_value>1e-300:
        ax_scatter1.text(0.05, 0.95, f'$p$={p_value_str}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
                 verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    else:
        ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
                 verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    
    
    slope_list.append(slope_type)
    intercept_list.append(intercept_type)






plt.show()





fig = plt.figure(figsize=(12, 20), dpi = 300)
grid = fig.add_gridspec(8, 5)
# Adjust layout: Increase vertical space between rows
plt.subplots_adjust(hspace=0.25)  # You can adjust this value (e.g., 0.5) to get more space

for i in range(40,78):
    
    Type = folder_means['Type'][i]
    i = i-40
    ax_scatter1 = fig.add_subplot(grid[i // 5, 
                                       i % 5])
    ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                        plot_data[plot_data["Type"] == Type]["SimOcean"], 
                        s = .5, color = palette[i])
    ax_scatter1.set_xlabel(Type)
    ax_scatter1.set_xticks([])
    ax_scatter1.set_yticks([])
    # Fit a line (linear fit in this case)
    p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                   plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
    y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
    # Perform linear regression using scipy to get p-value of the slope
    slope_type, intercept_type, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                        plot_data[plot_data["Type"] == Type]["SimOcean"])

    
    r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
    # Plot the fitted line
    ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
    N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])

    # Display the fitted formula in the bottom right corner
    formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, N={N:,}"
    ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    if p_value>1e-300:
        ax_scatter1.text(0.05, 0.95, f'$p$={p_value:.2e}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
                 verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    else:
        ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
                 verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    slope_list.append(slope_type)
    intercept_list.append(intercept_type)






mean_slope = np.mean(slope_list)
mean_intercept = np.mean(intercept_list)



slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data["SimSocial"], plot_data["SimOcean"])


diff_slope = slope-mean_slope
diff_intercept = intercept-mean_intercept


# Perform one-sample t-tests
t_stat_slope, p_value_slope = stats.ttest_1samp(slope_list, slope)
t_stat_intercept, p_value_intercept = stats.ttest_1samp(intercept_list, intercept)

# Output results
print(r"Slope t-test between: t={t_stat_slope:.3f}, $p$={p_value_slope:.2e}")
print(r"Intercept t-test result: t={t_stat_intercept:.3f}, $p$={p_value_intercept:.2e}")

plt.text(0.14, 0.35,
         f"mean_slope={mean_slope:.2f}, mean_intercept={mean_intercept:.2f}",
         horizontalalignment="left")
plt.text(0.14, 0.25,
         f"Slope $t$-test between individual SOSs and total:\nslope-mean_slope={diff_slope:.2e},\n$t$={t_stat_slope:.3f}, $p$={p_value_slope:.2e}",
         horizontalalignment="left")
plt.text(0.14, 0.15,
         f"Intercept $t$-test between individual SOSs and total:\nintercept-mean_intercept={diff_intercept:.2e},\n$t$={t_stat_intercept:.3f}, $p$={p_value_intercept:.2e}",
         horizontalalignment="left")


std_slope = np.std(slope_list)
std_intercept = np.std(intercept_list)
cv_slope = std_slope/mean_slope
cv_intercept = std_intercept/mean_intercept

plt.text(0.14, 0.05,
         f"Coefficient of variation:\nCV_slope={cv_slope:.2f}, CV_intercept={cv_intercept:.2f}",
         horizontalalignment="left")


plt.show()




