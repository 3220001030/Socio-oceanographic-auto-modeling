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
import scipy.stats as stats
from decimal import Decimal, getcontext

main_directory = '/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/webofscience'
# Initialize an empty dataframe
data = pd.DataFrame()

# Count total files to process for the progress bar
total_files = sum(len(files) for _, _, files in os.walk(main_directory) if files)

# Initialize tqdm progress bar with total files count
with tqdm(total=total_files, desc="Processing .xls files", unit="file") as pbar:
    # Loop through each folder in the main directory
    for folder_name in os.listdir(main_directory):
        folder_path = os.path.join(main_directory, folder_name)
        
        # Check if the path is indeed a directory
        if os.path.isdir(folder_path):
            # Loop through each .xls file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.xls'):
                    file_path = os.path.join(folder_path, file_name)
                    
                    # Read only the 'Abstract' column
                    temp_df = pd.read_excel(file_path, usecols=['Abstract'])
                    
                    # Add a new column for the folder name
                    temp_df['Folder'] = folder_name
                    
                    # Append the data to the main dataframe
                    data = pd.concat([data, temp_df], ignore_index=True)
                    
                    # Update progress bar
                    pbar.update(1)
data = data[data["Abstract"].notna()]
# Now `data` contains all the data from each .xls file, with an additional column for the folder name









# Word groups for comparison
ocean_words = [
               "Global"]
social_words = [
                "Pollutions"]


# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Preprocess word groups to get their embeddings
ocean_docs = [nlp(word) for word in ocean_words]
social_docs = [nlp(word) for word in social_words]



# Step 3: Classify New Text

# Load Excel file


# Extract "AB" column
names = data["Abstract"]
# Initialize vectors
SimOcean = []
SimSocial = []


# Iterate over names with progress bar
for name in tqdm(names, desc="Processing", unit="names", total=len(names)):
    # Create a SpaCy doc for the name
    doc = nlp(name)
    
    # Filter tokens to remove stopwords
    filtered_tokens = [token for token in doc if not token.is_stop]

    # Calculate average similarity with Ocean group
    ocean_similarities = [
        token.similarity(ocean_doc) for token in filtered_tokens for ocean_doc in ocean_docs
    ]
    avg_ocean_similarity = sum(ocean_similarities) / len(ocean_similarities) if ocean_similarities else 0


    # Calculate average similarity with Social group
    social_similarities = [
        token.similarity(social_doc) for token in filtered_tokens for social_doc in social_docs
    ]
    avg_social_similarity = sum(social_similarities) / len(social_similarities) if social_similarities else 0


    # Store the results
    SimOcean.append(avg_ocean_similarity)
    SimSocial.append(avg_social_similarity)














# Adjusting the plot axes to show one decimal place and setting custom ranges for the axes


xy = np.vstack([SimOcean, SimSocial])
z = gaussian_kde(xy)(xy)






# Combine into a DataFrame
output = pd.DataFrame({
    "SimSocial": SimSocial,
    "SimOcean": SimOcean,
    "z": z,
    "Type": data["Folder"]
})

# Save to CSV
output.to_csv("Plot.csv", index=False)








# Read back the data from the CSV
plot_data = pd.read_csv("Plot.csv")


fig = plt.figure(figsize=(12, 10), dpi = 600)
grid = fig.add_gridspec(3, 5, height_ratios=[4, 1, 1], width_ratios=[1, 1, 1, 1, 1])
# Adjust layout: Increase vertical space between rows
plt.subplots_adjust(hspace=0.25)  # You can adjust this value (e.g., 0.5) to get more space

plt.style.use('default')  # Ensure default style for a white background
# Main plot (first row, spans all columns)
ax_main = fig.add_subplot(grid[0, :])
ax_main.scatter(plot_data["SimSocial"], 
            plot_data["SimOcean"],
            c = plot_data["z"], alpha = .2,
            s=.5,cmap = 'Spectral',marker = 'o',
            label="First Scatter")



# Define and apply logarithmic normalization to color scale
norm = mcolors.LogNorm(vmin=1, vmax=1000)  # Adjust `vmin` and `vmax` as needed

# Adding a color bar as a legend for the density
sm = plt.cm.ScalarMappable(cmap="Spectral", norm=norm)
sm.set_array([])  # Empty array for matplotlib compatibility with ScalarMappable
cbar = plt.colorbar(sm, ax=ax_main, orientation='vertical', pad=0.02)
cbar.set_label('Log Density')

folder_means = plot_data.groupby('Type',
                            as_index=False)[['SimOcean', 'SimSocial']].mean()


wanted_left = [
          
          "Carbon sequestration",
          "Shipbuilding",
          "Aquaculture",
          "Ship operations",
          "Coastal infrastructure",
          
          "Tourist diving"
          "Freight shipping",
          "Land-based effluent",
          "Land reclamation",
          "Desalination",
          "Sediment extraction",
          "Marine mining",
          "Disaster displacement",
          "Mining waste disposal",
          "Coastal quarrying",
          "Trawling and seining",
          "Demersal seine netting",
          "Coastal defense",
          
          "Passenger ferries",
          "Freight shipping",
          "Tourist sports",]


wanted_right = ["Wastewater treatment",
                "Peel harvesting",
                "Oology",
                "Seaborne living",
                "Curio collecting",
                "Marine protected areas",
                "Port operations",
                "Nuclear stations",
                "Nautical tourism",
                "Line fishing",
                "Maritime security",
                "Maintenance dredging waste disposal",
                "Cultural heritage",
                "Hydrocarbon extraction",
                "Lagoon culvert",
                "Ship emissions",
                "Fishing traps"

                "Wind energy"]

wanted_top = ["Disaster recovery",]

filtered_left_folder_means = folder_means[folder_means["Type"].isin(wanted_left)]
filtered_right_folder_means = folder_means[folder_means["Type"].isin(wanted_right)]
filtered_top_folder_means = folder_means[folder_means["Type"].isin(wanted_top)]


# Annotate each folder's average point
for i, row in filtered_top_folder_means.iterrows():
    ax_main.text(row['SimSocial'], row['SimOcean'], row['Type'], fontsize=9, ha='left', va='top',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    

ax_main.scatter(filtered_top_folder_means['SimSocial'], 
            filtered_top_folder_means['SimOcean'], 
            color='black', s=10, zorder=3,
            label="Second Scatter1")  # Mark average points






# Annotate each folder's average point
for i, row in filtered_left_folder_means.iterrows():
    ax_main.text(row['SimSocial'], row['SimOcean'], row['Type'], fontsize=9, ha='left',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    

ax_main.scatter(filtered_left_folder_means['SimSocial'], 
            filtered_left_folder_means['SimOcean'], 
            color='black', s=10, zorder=3,
            label="Second Scatter1")  # Mark average points


# Annotate each folder's average point
for i, row in filtered_right_folder_means.iterrows():
    ax_main.text(row['SimSocial'], row['SimOcean'], row['Type'], fontsize=9, ha='right',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    

ax_main.scatter(filtered_right_folder_means['SimSocial'], 
            filtered_right_folder_means['SimOcean'], 
            color='black', s=10, zorder=3,
            label="Second Scatter2")  # Mark average points


# Add annotations with brackets
annotations = [
    {"text": "Resources\ninput", 
     "x": -0.003+0.0005/2, "left": -0.006+0.0005, "right": 0.000-0.0005},
    {"text": "Appropriation &\ncirculation", 
     "x": 0.000+0.0005+.009/2, "left": 0.000+0.0005, "right": 0.01-0.0005},
    {"text": "Transformation &\nconservation", 
     "x": 0.01+0.0005+0.009/2, "left": 0.01+0.0005, "right": 0.02-0.0005},
    {"text": "Consumption &\nexcretion", 
     "x": 0.02+0.0005+0.009/2, "left": 0.02+0.0005, "right": 0.03-0.0005},
    {"text": "Wastes\noutput", 
     "x": 0.03+0.0005+0.009/2, "left": 0.03+0.0005, "right": 0.040-0.0005}
]

for annotation in annotations:
    x_pos = annotation["x"]
    left = annotation["left"]
    right = annotation["right"]
    
    # Add the annotation text
    ax_main.text(x_pos, 0.166, annotation["text"], ha="center", va="top", fontsize=10, color="black")
    
    # Draw bracket using a curve or line
    ax_main.plot(
        [left, left, right, right],  # X coordinates
        [0.159, 0.16, 0.16, 0.159],  # Y coordinates
        color='black', linewidth=1, clip_on=False
    )



# Add annotations with brackets
annotations = [
    {"text": "Marine\nspecies", 
     "y": (0.17+0.18)/2+0.005-0.003, "left": 0.17+0.0005-0.003, "right": 0.18-0.0005-0.003},
    {"text": "Aquatic\nenvironment", 
     "y": (0.18+0.2-0.003)/2+0.01, "left": 0.18+0.0005-0.003, "right": 0.2-0.0005},
    {"text": "Macro\nConditions", 
     "y": (0.2+0.215)/2+0.008, "left": 0.2+0.0005, "right": 0.215-0.0005},
    {"text": "Regional\nseas", 
     "y": (0.215+0.23)/2+0.006, "left": 0.215+0.0005, "right": 0.23-0.0005},
    {"text": "Global\noceans", 
     "y": (0.23+0.245)/2+0.006, "left": 0.23+0.0005, "right": 0.245}
]

for annotation in annotations:
    y_pos = annotation["y"]
    left = annotation["left"]
    right = annotation["right"]
    
    # Add the annotation text
    ax_main.text(-0.007, y_pos, annotation["text"], ha="center", va="top", fontsize=10, color="black", rotation=90)
    
    # Draw bracket using a curve or line
    ax_main.plot(
        [-0.01+0.001, -0.01+0.0015, -0.01+0.0015, -0.01+0.001],  # X coordinates
        [left, left, right, right],  # Y coordinates
        color='black', linewidth=1, clip_on=False
    )




# Set axis labels with one decimal place and adjust the range
ax_main.set_xlabel('Position in sociometabolic cycle', 
                   fontsize=14)
ax_main.set_ylabel('Oceanographic scale',
                   fontsize=14)


ax_main.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # Format x-axis
ax_main.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))  # Format y-axis


# Set the x and y limits
ax_main.set_xlim(-0.01, 0.04)  # Set x-axis limits from 2 to 8
ax_main.set_ylim(0.157, 0.247)  # Set y-axis limits from -0.5 to 1



























ax_scatter1 = fig.add_subplot(grid[1, 0])
ax_scatter1.scatter(plot_data["SimSocial"], 
                    plot_data["SimOcean"],
                    c = plot_data["z"],
                    s = .5, cmap = 'Spectral')
ax_scatter1.set_xlabel("All")
ax_scatter1.set_xticks([])
ax_scatter1.set_yticks([])
# Fit a line (linear fit in this case)
p = np.polyfit(plot_data["SimSocial"], 
               plot_data["SimOcean"], 1)  # Linear fit (degree 1)
y_fit = np.polyval(p, plot_data["SimSocial"])  # Evaluate the fitted line
r2 = r2_score(plot_data["SimOcean"], y_fit)

# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data["SimSocial"], plot_data["SimOcean"])

# Plot the fitted line
ax_scatter1.plot(plot_data["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
N = len(plot_data["SimSocial"])    
# Display the fitted formula in the bottom right corner
formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, $n$={N:,}"
ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
         verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# Get the first color from the 'Set1' colormap
cmap = plt.cm.Set1
first_color = cmap(0)  # 0 corresponds to the first color in the colormap










Type = "Aquaculture"
ax_scatter1 = fig.add_subplot(grid[1, 1])
ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"], 
                    s = .5, color = cmap(0))
ax_scatter1.set_xlabel(Type)
ax_scatter1.set_xticks([])
ax_scatter1.set_yticks([])
# Fit a line (linear fit in this case)
p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
               plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"])

# Plot the fitted line
ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])

# Display the fitted formula in the bottom right corner
formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, $n$={N:,}"
ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
if p_value<1e-300:
    ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
else:
    ax_scatter1.text(0.05, 0.95, f'$p$={p_value:.0e}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))










Type = "Marine protected areas"
ax_scatter1 = fig.add_subplot(grid[1, 2])
ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"], 
                    s = .5, color = cmap(1))
ax_scatter1.set_xlabel(Type)
ax_scatter1.set_xticks([])
ax_scatter1.set_yticks([])
# Fit a line (linear fit in this case)
p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
               plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"])

# Plot the fitted line
ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])
# Display the fitted formula in the bottom right corner
formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, $n$={N:,}"
ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
if p_value<1e-300:
    ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
else:
    ax_scatter1.text(0.05, 0.95, f'$p$={p_value:.0e}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))











Type = "Desalination"
ax_scatter1 = fig.add_subplot(grid[1, 3])
ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"], 
                    s = .5, color = cmap(2))
ax_scatter1.set_xlabel(Type)
ax_scatter1.set_xticks([])
ax_scatter1.set_yticks([])
# Fit a line (linear fit in this case)
p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
               plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"])

# Plot the fitted line
ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])

# Display the fitted formula in the bottom right corner
formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, $n$={N:,}"
ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
if p_value<1e-300:
    ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
else:
    ax_scatter1.text(0.05, 0.95, f'$p$={p_value:.0e}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))








Type = "Land reclamation"
ax_scatter1 = fig.add_subplot(grid[2, 0])
ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"], 
                    s = .5, color = cmap(3))
ax_scatter1.set_xlabel(Type)
ax_scatter1.set_xticks([])
ax_scatter1.set_yticks([])
# Fit a line (linear fit in this case)
p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
               plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"])

# Plot the fitted line
ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])
 
# Display the fitted formula in the bottom right corner
formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, $n$={N:,}"
ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
if p_value<1e-300:
    ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
else:
    ax_scatter1.text(0.05, 0.95, f'$p$={p_value:.0e}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))





Type = "Disaster recovery"
ax_scatter1 = fig.add_subplot(grid[2, 1])
ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"], 
                    s = .5, color = cmap(4))
ax_scatter1.set_xlabel(Type)
ax_scatter1.set_xticks([])
ax_scatter1.set_yticks([])
# Fit a line (linear fit in this case)
p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
               plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"])

# Plot the fitted line
ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])

# Display the fitted formula in the bottom right corner
formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, $n$={N:,}"
ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
if p_value<1e-300:
    ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
else:
    ax_scatter1.text(0.05, 0.95, f'$p$={p_value:.0e}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))



Type = "Marine mining"
ax_scatter1 = fig.add_subplot(grid[2, 2])
ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"], 
                    s = .5, color = cmap(7))
ax_scatter1.set_xlabel(Type)
ax_scatter1.set_xticks([])
ax_scatter1.set_yticks([])
# Fit a line (linear fit in this case)
p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
               plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"])

# Plot the fitted line
ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])

# Display the fitted formula in the bottom right corner
formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, $n$={N:,}"
ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
if p_value<1e-300:
    ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
else:
    ax_scatter1.text(0.05, 0.95, f'$p$={p_value:.0e}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))



Type = "Ship operations"
ax_scatter1 = fig.add_subplot(grid[2, 3])
ax_scatter1.scatter(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"], 
                    s = .5, color = cmap(6))
ax_scatter1.set_xlabel(Type)
ax_scatter1.set_xticks([])
ax_scatter1.set_yticks([])
# Fit a line (linear fit in this case)
p = np.polyfit(plot_data[plot_data["Type"] == Type]["SimSocial"], 
               plot_data[plot_data["Type"] == Type]["SimOcean"], 1)  # Linear fit (degree 1)
y_fit = np.polyval(p, plot_data[plot_data["Type"] == Type]["SimSocial"])  # Evaluate the fitted line
r2 = r2_score(plot_data[plot_data["Type"] == Type]["SimOcean"], y_fit)
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value, std_err = stats.linregress(plot_data[plot_data["Type"] == Type]["SimSocial"], 
                    plot_data[plot_data["Type"] == Type]["SimOcean"])

# Plot the fitted line
ax_scatter1.plot(plot_data[plot_data["Type"] == Type]["SimSocial"], y_fit, color="black", label=f"Fit: y = {p[0]:.2f}x + {p[1]:.2f}")
N = len(plot_data[plot_data["Type"] == Type]["SimSocial"])

# Display the fitted formula in the bottom right corner
formula = f"y = {p[0]:.2f}x + {p[1]:.2f}\n$R^2 = {r2:.2f}$, $n$={N:,}"
ax_scatter1.text(0.95, 0.05, formula, transform=ax_scatter1.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
if p_value<1e-300:
    ax_scatter1.text(0.05, 0.95, f'$p$<1e-300', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
else:
    ax_scatter1.text(0.05, 0.95, f'$p$={p_value:.0e}', transform=ax_scatter1.transAxes, fontsize=10, color='black', 
             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))






# Save the plot as a PDF
jpeg_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Fig2.jpeg"
plt.savefig(jpeg_file, format="jpeg", bbox_inches="tight")  # Save as PDF with tight layout

plt.show()


















type_counts = plot_data['Type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']


# Initialize groups as empty DataFrames
num_groups = 5
groups = [pd.DataFrame(columns=['Type', 'Count']) for _ in range(num_groups)]

# Assign each Type to a group in a round-robin descending order
for i, row in type_counts.iterrows():
    group_index = i % num_groups
    groups[group_index] = pd.concat([groups[group_index], pd.DataFrame([row])], ignore_index=True)

# Assign each group's DataFrame to a variable
group_1_df = groups[0]
group_2_df = groups[1]
group_3_df = groups[2]
group_4_df = groups[3]
group_5_df = groups[4]

# Now, group_1_df, group_2_df, group_3_df, group_4_df, and group_5_df are your resulting DataFrames



slope, intercept, r_value, p_value, std_err = stats.linregress(folder_means["SimSocial"], 
                    folder_means["SimOcean"])




