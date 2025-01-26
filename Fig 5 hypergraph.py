import os
import pandas as pd
import numpy as np
import networkx as nx
import hypergraphx as hgx
from hypergraphx.representations.projections import clique_projection
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import hypergraphx as hgx
from hypergraphx.motifs import compute_motifs
from hypergraphx.readwrite import load_hypergraph
from hypergraphx.viz import plot_motifs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgi
import networkx as nx
import hypergraphx as hgx
import numpy as np
import csv
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
dimensions = pd.read_csv("/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/webofscience/type_and_slope_output.csv")

# Define the directory where your CSV and XLSX files are stored
directory = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/矩阵数据/"  # Replace with your actual directory path
n = 4
total_nodes = 0
total_edges = 0
total_hyperedges = 0


# Function to load CSV files
def load_csv(file_path):
    df = pd.read_csv(file_path, header=None)  # Assume the first column is the row index
    df = df.fillna(0)
    system = df.iloc[0, 0]
    # Remove the first row and the first column
    df = df.drop(0, axis=0)  # Drop the first row
    df = df.drop(0, axis=1)  # Drop the first column
    
    return {"system": system, "matrix": df}



# Load all CSV and XLSX files in the directory and subdirectories
def load_data_from_directory(directory):
    matrices = {}
    
    # Traverse all subdirectories and files in the directory
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            # Load CSV files
            if file_name.endswith(".csv"):
                print(f"Loading CSV file: {file_path}")
                matrix = load_csv(file_path)
                matrices[file_path] = matrix
            
    return matrices


# Find consecutive sequences of nodes (paths of length >= n)
def find_consecutive_hyperedges(adj_matrix, n):
    num_nodes = min(adj_matrix.shape[0], adj_matrix.shape[1])
    hyperedges = set()
    

    # Loop through all nodes to find paths of length >= n
    for start_node in range(num_nodes):
        visited = set()
        stack = [(start_node, [start_node])]  # Stack for DFS: (current node, path)

        while stack:
            current, path = stack.pop()
            visited.add(current)

            # If the path length is >= n, add it as a hyperedge
            if len(path) >= n:
                hyperedges.add(tuple(path))

            # Visit neighbors of the current node
            for neighbor in range(num_nodes):
                if adj_matrix[current, neighbor] > 0 and neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))

    return hyperedges



def hypergraph_statistics(adj_matrix, system, x):
    # Create an empty hypergraph
    h = hgx.Hypergraph()
    n = round(dimensions[dimensions["Type"].str.contains(system, case=False, na=False)]["Slope"]).iloc[0]
    num_hyperedges = 0
    
    adj_matrix = adj_matrix.astype(float)
    # Calculate the number of nodes from the adjacency matrix
    num_nodes = min(adj_matrix.shape[0], adj_matrix.shape[1])
    

    # Add nodes to the hypergraph
    h.add_nodes(list(range(1, num_nodes + 1)))
    
    adj_matrix = adj_matrix.to_numpy()

    # Iterate over the adjacency matrix to add edges to the hypergraph
    for i in range(num_nodes):
        # Find non-zero entries in row i (i.e., connected nodes)
        row_connected_nodes = np.where(adj_matrix[i] != 0)[0] + 1  # +1 to match node numbering
        
        # Find non-zero entries in column i (i.e., connected nodes)
        col_connected_nodes = np.where(adj_matrix[:, i] != 0)[0] + 1  # +1 to match node numbering
        


        
        for j in range(num_nodes):  # Only check upper triangle to avoid duplicates
            if adj_matrix[i, j] > 0:
                h.add_edges([(i + 1, j + 1)])



        # 🟥🟥🟥超边机制1：Center degrees🟥🟥🟥
        # Check if the number of connections in row i exceeds n (Hyperedge)
        if len(row_connected_nodes) >= n-1:
            h.add_edges([tuple(row_connected_nodes)])
            num_hyperedges += 1

        # Check if the number of connections in column i exceeds n (Hyperedge)
        if len(col_connected_nodes) >= n-1:
            h.add_edges([tuple(col_connected_nodes)])
            num_hyperedges += 1
        
    # 🟥🟥🟥超边机制2：Path length🟥🟥🟥
    # Find and add hyperedges
    consecutive_hyperedges = find_consecutive_hyperedges(adj_matrix, n-1)
    for hyperedge in consecutive_hyperedges:
        h.add_edge(hyperedge)
        num_hyperedges += 1




    motifs = compute_motifs(h, order=x, runs_config_model=10)
    motif_profile = [i[1] for i in motifs['norm_delta']]
    
    
            
    H = xgi.Hypergraph()
    H.add_edges_from(h.get_edges())
    centers, heights = xgi.degree_histogram(H)
    
    
    
    n_nodes = adj_matrix.shape[0]
    
    n_edges = adj_matrix.sum()
    
    
    return motif_profile, motifs['norm_delta'], centers, heights, n_nodes, n_edges, num_hyperedges




# Example of using the function
matrices = load_data_from_directory(directory)

# Step 1: Define the system groups manually (you can modify this based on your actual groupings)
system_groups = {
    'Resources input': ['Marine education',

'Line fishing',
'Peel harvesting',
'Tourist sports',

'Oology',
'Recreational fishing',
'Trawling and seining',
'Demersal seine netting',
'Curio collecting',
'Wind energy',
'Wave energy',
'Shellfish harvesting',
'Consumptive fishing',
'Fishing traps',],
    
    'Appropriation & circulation': ['Port operations',
'Cable maintenance',
'Cruise tourism',
'Waterway transport',
'Drift netting',
'Shipbuilding',
'Bait digging',
'Maritime security',
'Cable laying',
'Survey and monitoring',

'Passenger ferries',

'Nautical tourism',
'Harbor facilities',
'Tourist diving',

'Freight shipping',
'Static fishing',
'Marine protected areas',],
    
    'Transformation & conservation': ['Marine research',
'Seaside resorts',
'Seaweed harvesting',
'Cultural heritage',

'Ship operations',
'Hydraulic dredging',
'Beach tourism',
'Bivalve gathering',
'Seaborne living',
'Bioprospecting',

'Coastal quarrying',
'Pipeline maintenance',
'Maerl aggregation',
'Coastal infrastructure',
'Marine mining',
'Pipeline construction',
'Aquaculture',
'Coastal defense',
'Tidal energy',
'Disaster recovery',

'Disaster assistance',
'Lagoon culvert',
'Nuclear stations',
'Desalination',
'Disaster displacement',],
    
    'Consumption & excretion': ['Power stations',
'Sediment extraction',
'Military debris',
'Ship emissions',
'Ship emission',
'Hydrocarbon extraction',
'Capital dredging',
'Land reclamation',
'Quarrying waste disposal',
'Maintenance dredging',
'Inorganic mining','Waste gas emission',
'Nuclear wastewater',
'Power plant wastewater',
'Mining waste disposal',
'Capital dredging waste disposal',],
    
    'Wastes output': [
'Carbon sequestration',
'Urban wastewater',
'Land-based effluent',
'Industrial pollution',
'Waste and spoil aggregation',
'Wastewater treatment',
'Maintenance dredging waste disposal',],
}

# Define your group colors using hex color codes (one-cell hex)
group_colors = {
    "Resources input": "#3b75af",  # Blue
    "Appropriation & circulation": "#ef8636",  # Orange
    "Transformation & conservation": "#519e3e",  # Green
    "Consumption & excretion": "#c53a32",
    "Wastes output": "#8d69b8",
    # Add other group colors as needed
}




























# 🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥Figure 1🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥

list_of_lists = []
norms = []
hist = []
# Initialize a dictionary to store data
scaling = {
    "Nodes": [],
    "Edges": [],
    "Hyperedges": []
}

# Display the loaded matrices
for file_path, matrix_info in matrices.items():
    adj_matrix = matrix_info["matrix"]
    system = matrix_info["system"]
    print("\n\n\n")
    print(file_path)
    print(system)
    new_list, norm, centers, heights, num_nodes, num_edges, num_hyperedges = hypergraph_statistics(adj_matrix, system, 3)
    
    list_of_lists.append((new_list, system))
    norms.append(norm)
    hist.append((centers, heights, system))
    
    total_nodes += num_nodes
    total_edges += num_edges
    total_hyperedges += num_hyperedges
    scaling["Nodes"].append(num_nodes)
    scaling["Edges"].append(num_edges)
    scaling["Hyperedges"].append(num_hyperedges)
    

# Step 2: Group lists by system
grouped_lists = {}
for new_list, system in list_of_lists:
    if system not in grouped_lists:
        grouped_lists[system] = []
    grouped_lists[system].append(new_list)

# Step 3: Now group by system group
grouped_by_system_group = {}

for system, system_list in grouped_lists.items():
    # Find the system group for the current system
    for group, systems in system_groups.items():
        if system in systems:
            if group not in grouped_by_system_group:
                grouped_by_system_group[group] = []
            grouped_by_system_group[group].extend(system_list)

# Step 4: Calculate the average for each system group
average_lists_by_group = {group: np.mean(group_list, axis=0) for group, group_list in grouped_by_system_group.items()}
group1 = average_lists_by_group
# Step 4: Use Seaborn's built-in color palette to automatically assign colors
# Here we use 'Set1' for color palette, but you can choose any Seaborn palette
group_palette = sns.color_palette("Set1", len(grouped_by_system_group))  # Get palette for the number of groups
group_colors = dict(zip(grouped_by_system_group.keys(), group_palette))  # Map groups to colors

# Step 1: Choose an inbuilt color palette
palette = sns.color_palette("deep", n_colors=len(grouped_by_system_group))


# Step 5: Create row colors for the clustermap based on the groupings
row_colors = []
for group, group_list in grouped_by_system_group.items():
    color = group_colors[group]  # Get the color for this group
    row_colors.extend([color] * len(group_list))  # Repeat the color for each sample in the group

# Convert row_colors to a NumPy array (removes any index)
row_colors = np.array(row_colors)

# Now we can proceed with the plotting code as before


plt.style.use('default')  # Ensure default style for a white background
# Main plot (first row, spans all columns)
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)  # You can adjust this value (e.g., 0.5) to get more space


fig, ax = plt.subplots(figsize=(10, 6), dpi = 600)
















# Create the plot and store the handles and labels for later sorting
handles = []
labels = []


for group, avg_lst in average_lists_by_group.items():
    group_list = grouped_by_system_group[group]
    list_count = len(grouped_by_system_group[group])  # Count the number of lists in the group
    line, = ax.plot(avg_lst, label=group,
                     color=group_colors[group],
                     linewidth=3)  # Plot the average list for the group (without count in label)
    handles.append(line)
    labels.append(group)  # Store the group name

ax.set_ylim(-1, 1)
# Get the number of x-ticks (positions) in the plot
num_xticks = len(ax.get_xticks())

# Generate x-tick labels starting from 1
xticks_labels = [str(i) for i in range(num_xticks)]

# Set the x-ticks labels starting from 1
ax.set_xticklabels(xticks_labels)

















# Desired order for the legend (adjust as needed)
preferred_order = ['Resources input', 
                   'Appropriation & circulation', 
                   'Transformation & conservation',
                   'Consumption & excretion',
                   'Wastes output']

handles, labels = plt.gca().get_legend_handles_labels()

# Sort handles and labels according to the preferred order (using group name)
sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: preferred_order.index(x[0]))
sorted_labels, sorted_handles = zip(*sorted_handles_labels)

# Add the legend with the custom order and updated labels (including the counts)
legend_labels = [f"{label} ($n$={len(grouped_by_system_group[label])})" for label in sorted_labels]

# Add the legend with the custom labels and order
plt.legend(sorted_handles, legend_labels, fontsize=16,
           frameon=False, loc='upper left',
           labelspacing=0.5)

# Labels for the axes
plt.xlabel("Hypergraph motif", fontsize=18)
plt.ylabel("Abundance (Δ)", fontsize=18)
# Remove the upper and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Add horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='--', linewidth=3)
# Increase the font size for x and y ticks
plt.tick_params(axis='both', which='major', labelsize=18)














# Create a hypergraph
hyperedges = [[1, 2], [1,2,3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.1, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well



# Create a hypergraph
hyperedges = [[1, 2], [1,2,3], [1, 3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.236, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well




# Create a hypergraph
hyperedges = [[1, 2], [1,2,3], [1,3], [2,3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.372, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well



# Create a hypergraph
hyperedges = [[1, 2], [1,3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.508, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well


# Create a hypergraph
hyperedges = [[1, 2], [1,3], [2,3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.644, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well


# Create a hypergraph
hyperedges = [[1, 2, 3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.78, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well

fig.savefig("f1.png", format="png", bbox_inches='tight')
plt.close(fig)  # Close the figure to avoid it appearing on the screen
img1 = plt.imread("f1.png")

































# 🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥Figure 2🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥
average_lists = {system: np.mean(group, axis=0) for system, group in grouped_lists.items()}
# Step 2: Capitalize the first letter of each system in average_lists
average_lists = {system.capitalize(): data for system, data in average_lists.items()}


# Initialize lists to hold data
systems = []
groups = []
colors = []
average_lists_data = []

# Loop through each system in average_lists
for system, avg_list in average_lists.items():
    system_matched = False  # Flag to check if system is matched to a color
    for group, systems_in_group in system_groups.items():
        if system in systems_in_group:
            # If the system is in this group, append the corresponding values
            systems.append(system)
            groups.append(group)
            colors.append(group_colors[group])  # Get the color for the group
            average_lists_data.append(avg_list)  # Get the average list for the system
            system_matched = True
            break  # Stop checking other groups once a match is found
    
    if not system_matched:
        print(f"System '{system}' is missing a match in group_colors!")

# Now, create a DataFrame with the data
df = pd.DataFrame({
    'System': systems,
    'Group': groups,
    'Color': colors,
    'Average List': average_lists_data
})




# Convert the dictionary into a list of lists for correlation matrix calculation
heatmap_data = np.array(df['Average List'].tolist())


# Calculate the correlation matrix (using Pearson correlation)
correlation_matrix = np.corrcoef(heatmap_data)

# Perform hierarchical clustering on the correlation matrix
# `linkage` performs the clustering and returns a linkage matrix
Z = linkage(correlation_matrix, method='ward')

# Plot the heatmap with clustering
fig = plt.figure(figsize=(10, 6), dpi = 600)
# Step 2: Create row_colors list based on the system's group
# Step 2: Create row_colors list based on system's group affiliation
# Step 1: Generate row_colors for each system in average_lists based on system_groups


# Step 3: Convert average_lists to a NumPy array for use in clustermap
data_for_clustermap = np.array(list(average_lists.values()))
sns.set(font_scale=2)
# Create a clustermap with hierarchical clustering
g1 = sns.clustermap(correlation_matrix, cmap="coolwarm", annot=False, 
               figsize=(10, 6), 
               yticklabels=False, xticklabels=False,
               row_cluster=True, col_cluster=True, method='ward',
                
               cbar_pos=(0, .7, .04, .3),
               dendrogram_ratio=(.2, .1),
               row_colors=df["Color"].to_numpy(),
               tree_kws={'linewidths': 2})
g1.fig.savefig("g1.png", format="png", bbox_inches='tight')
plt.close(g1.fig)  # Close the figure to avoid it appearing on the screen

# Display the heatmap
plt.show()











# 🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥Figure 3🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥

list_of_lists = []
norm4 = []

# Display the loaded matrices
for file_path, matrix_info in matrices.items():
    adj_matrix = matrix_info["matrix"]
    system = matrix_info["system"]
    print("\n\n\n")
    print(file_path)
    print(system)
    new_list, norm, centers, heights, num_nodes, num_edges, num_hyperedges = hypergraph_statistics(adj_matrix, system, 4)
    list_of_lists.append((new_list, system))
    norm4.append(norm)

# Step 2: Group lists by system
grouped_lists = {}
for new_list, system in list_of_lists:
    if system not in grouped_lists:
        grouped_lists[system] = []
    grouped_lists[system].append(new_list)

# Step 3: Now group by system group
grouped_by_system_group = {}

for system, system_list in grouped_lists.items():
    # Find the system group for the current system
    for group, systems in system_groups.items():
        if system in systems:
            if group not in grouped_by_system_group:
                grouped_by_system_group[group] = []
            grouped_by_system_group[group].extend(system_list)

# Step 4: Calculate the average for each system group
average_lists_by_group = {group: np.mean(group_list, axis=0) for group, group_list in grouped_by_system_group.items()}
group2 = average_lists_by_group
# Step 4: Use Seaborn's built-in color palette to automatically assign colors
# Here we use 'Set1' for color palette, but you can choose any Seaborn palette
group_palette = sns.color_palette("Set1", len(grouped_by_system_group))  # Get palette for the number of groups
group_colors = dict(zip(grouped_by_system_group.keys(), group_palette))  # Map groups to colors

# Step 1: Choose an inbuilt color palette
palette = sns.color_palette("deep", n_colors=len(grouped_by_system_group))


# Step 5: Create row colors for the clustermap based on the groupings
row_colors = []
for group, group_list in grouped_by_system_group.items():
    color = group_colors[group]  # Get the color for this group
    row_colors.extend([color] * len(group_list))  # Repeat the color for each sample in the group

# Convert row_colors to a NumPy array (removes any index)
row_colors = np.array(row_colors)

# Now we can proceed with the plotting code as before


plt.style.use('default')  # Ensure default style for a white background
# Main plot (first row, spans all columns)
plt.subplots_adjust(bottom=0.1, left=0.1, right=0.9, top=0.9)  # You can adjust this value (e.g., 0.5) to get more space


fig, ax = plt.subplots(figsize=(10, 6), dpi = 600)
















# Create the plot and store the handles and labels for later sorting
handles = []
labels = []

for group, avg_lst in average_lists_by_group.items():
    group_list = grouped_by_system_group[group]
    list_count = len(grouped_by_system_group[group])  # Count the number of lists in the group
    line, = ax.plot(avg_lst, label=group,
                     color=group_colors[group],
                     linewidth=2)  # Plot the average list for the group (without count in label)
    handles.append(line)
    labels.append(group)  # Store the group name

ax.set_ylim(-0.4, 0.3)
# Get the number of x-ticks (positions) in the plot


















# Desired order for the legend (adjust as needed)
preferred_order = ['Resources input', 
                   'Appropriation & circulation', 
                   'Transformation & conservation',
                   'Consumption & excretion',
                   'Wastes output']

handles, labels = plt.gca().get_legend_handles_labels()

# Sort handles and labels according to the preferred order (using group name)
sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: preferred_order.index(x[0]))
sorted_labels, sorted_handles = zip(*sorted_handles_labels)

# Add the legend with the custom order and updated labels (including the counts)
legend_labels = [f"{label} ($n$={len(grouped_by_system_group[label])})" for label in sorted_labels]



# Labels for the axes
plt.xlabel("Hypergraph motif", fontsize=18)
plt.ylabel("Abundance (Δ)", fontsize=18)
# Remove the upper and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# Add horizontal line at y=0
plt.axhline(y=0, color='black', linestyle='--', linewidth=3)
# Increase the font size for x and y ticks
plt.tick_params(axis='both', which='major', labelsize=18)














# Create a hypergraph
hyperedges = [[1, 2], [1, 2, 3], [1, 2, 3, 4]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.1, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well



# Create a hypergraph
hyperedges = [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 4], [2, 3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.2, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well




# Create a hypergraph
hyperedges = [[1, 2], [1, 2, 3], [1, 2, 4], [1, 3], [1, 3, 4], [1, 4], [2, 3], [2, 3, 4]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.3, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well



# Create a hypergraph
hyperedges = [[1, 2], [1, 2, 3], [1, 3], [1, 4], [2, 3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.4, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well


# Create a hypergraph
hyperedges = [[1, 2], [1, 2, 3, 4], [1, 3], [1, 3, 4], [1, 4], [2, 3], [2, 3, 4], [2, 4]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.5, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot

# Adjust layout to avoid overlap and ensure everything fits well


# Create a hypergraph
hyperedges = [[1, 2], [1, 2, 4], [1, 3], [2, 3, 4]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.6, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot


# Create a hypergraph
hyperedges = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 4], [1, 4], [2, 3]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.7, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot



# Create a hypergraph
hyperedges = [[1, 3], [1, 4], [2, 3], [2, 4]]
H = xgi.Hypergraph(hyperedges)
# Create an axis for the hypergraph plot below the main plot
# Lower the bottom position to place it below the main plot
ax_hypergraph = fig.add_axes([0.8, 0.15, 0.15, 0.15])  # Lowered bottom to give space

# Define the layout for the hypergraph
pos = xgi.circular_layout(H)

# Draw the hypergraph on the specific axes
xgi.draw(H, pos=pos, hull=True,
         node_labels=True,
         hyperedge_labels=False,
         node_size=11,
         node_fc="#afd2d0",
         node_lw=1,
         dyad_lw=3,
         ax=ax_hypergraph)  # Embed the hypergraph below the main plot




# Adjust layout to avoid overlap and ensure everything fits well

fig.savefig("f2.png", format="png", bbox_inches='tight')
plt.close(fig)  # Close the figure to avoid it appearing on the screen
img1 = plt.imread("f2.png")

















# 🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥Figure 4🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥
average_lists = {system: np.mean(group, axis=0) for system, group in grouped_lists.items()}
# Step 2: Capitalize the first letter of each system in average_lists
average_lists = {system.capitalize(): data for system, data in average_lists.items()}


# Initialize lists to hold data
systems = []
groups = []
colors = []
average_lists_data = []

# Loop through each system in average_lists
for system, avg_list in average_lists.items():
    system_matched = False  # Flag to check if system is matched to a color
    for group, systems_in_group in system_groups.items():
        if system in systems_in_group:
            # If the system is in this group, append the corresponding values
            systems.append(system)
            groups.append(group)
            colors.append(group_colors[group])  # Get the color for the group
            average_lists_data.append(avg_list)  # Get the average list for the system
            system_matched = True
            break  # Stop checking other groups once a match is found
    
    if not system_matched:
        print(f"System '{system}' is missing a match in group_colors!")

# Now, create a DataFrame with the data
df = pd.DataFrame({
    'System': systems,
    'Group': groups,
    'Color': colors,
    'Average List': average_lists_data
})




# Convert the dictionary into a list of lists for correlation matrix calculation
heatmap_data = np.array(df['Average List'].tolist())


# Calculate the correlation matrix (using Pearson correlation)
correlation_matrix = np.corrcoef(heatmap_data)

# Perform hierarchical clustering on the correlation matrix
# `linkage` performs the clustering and returns a linkage matrix
Z = linkage(correlation_matrix, method='ward')

# Plot the heatmap with clustering
fig = plt.figure(figsize=(10, 6), dpi = 600)
# Step 2: Create row_colors list based on the system's group
# Step 2: Create row_colors list based on system's group affiliation
# Step 1: Generate row_colors for each system in average_lists based on system_groups


# Step 3: Convert average_lists to a NumPy array for use in clustermap
data_for_clustermap = np.array(list(average_lists.values()))
sns.set(font_scale=2)
# Create a clustermap with hierarchical clustering
g2 = sns.clustermap(correlation_matrix, cmap="coolwarm", annot=False, 
               figsize=(10, 6), 
               yticklabels=False, xticklabels=False,
               row_cluster=True, col_cluster=True, method='ward',
                
               cbar_pos=(0, .7, .04, .3),
               dendrogram_ratio=(.2, .1),
               row_colors=df["Color"].to_numpy(),
               tree_kws={'linewidths': 2})
g2.fig.savefig("g2.png", format="png", bbox_inches='tight')
plt.close(g1.fig)  # Close the figure to avoid it appearing on the screen

# Display the heatmap
plt.show()































# 🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥Final Figure 🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥
# Create a 3x2 subplot grid
fig, axs = plt.subplots(3, 2, figsize=(15, 12.5), dpi = 600,
                        height_ratios=[2,2,.9],
                        constrained_layout=True)  # You can adjust figsize based on your preferences
plt.style.use('default')
# Load and display the images on the specified subplot positions
# ax[0, 0] for f1.png and ax[0, 1] for g1.png
img_f1 = plt.imread('/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/f1.png')
img_g1 = plt.imread('/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/g1.png')
img_f2 = plt.imread('/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/f2.png')
img_g2 = plt.imread('/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/g2.png')
# Plot f1.png in position (0, 0)
axs[0, 0].imshow(img_f1)
axs[0, 0].axis('off')  # Optionally turn off axis

# Plot g1.png in position (0, 1)
axs[0, 1].imshow(img_g1)
axs[0, 1].axis('off')  # Optionally turn off axis

# Plot f1.png in position (0, 0)
axs[1, 0].imshow(img_f2)
axs[1, 0].axis('off')  # Optionally turn off axis

# Plot g1.png in position (0, 1)
axs[1, 1].imshow(img_g2)
axs[1, 1].axis('off')  # Optionally turn off axis

# Set the titles for each subplot with font size 14
axs[0, 0].set_title(r"$\mathbf{{A}}$ 3rd order hypergraph motif significance profiles", 
                    fontsize=18, loc='left')
axs[0, 1].set_title(r"$\mathbf{{B}}$ 3rd order hypergraph motif correlation matrix", 
                    fontsize=18, loc='left')
axs[1, 0].set_title(r"$\mathbf{{C}}$ 4th order hypergraph motif significance profiles", 
                    fontsize=18, loc='left')
axs[1, 1].set_title(r"$\mathbf{{D}}$ 4th order hypergraph motif correlation matrix", 
                    fontsize=18, loc='left')


# Adjust layout to ensure no overlap
plt.tight_layout()
# Adjust layout to ensure no overlap and set minimal space
plt.subplots_adjust(wspace=0, hspace=0)


axs[2, 0].axis('off') 
axs[2, 1].axis('off') 
axs[2, 0].set_title(r"$\mathbf{{E}}$ Average hyperedge degree distribution", 
                    fontsize=18, loc='left')


ax1 = fig.add_axes([0.06, 0.0, 0.15, 0.18])
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# Increase the font size of the tick labels using set_xticklabels and set_yticklabels
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_title("Resources input", fontsize=13)

# Step 1: Filter histograms whose systems belong to the "Resources input" group
filtered_hist = [
    (centers, heights, system) for centers, heights, system in hist if system in system_groups['Resources input']
]

# Step 2: Merge centers and heights
merged_centers = []
merged_heights = []

for centers, heights, system in filtered_hist:
    merged_centers.extend(centers)
    merged_heights.extend(heights)

# Step 3: Find unique centers and calculate the average height for each center
unique_centers = sorted(set(merged_centers))

average_centers = []
average_heights = []

# Calculate average heights using the length of filtered_hist
num_filtered_hist = len(filtered_hist)

for center in unique_centers:
    # Find all heights corresponding to the current center
    heights_for_center = [height for c, height in zip(merged_centers, merged_heights) if c == center]
    
    # Sum all heights and divide by the total number of entries in filtered_hist
    avg_height = sum(heights_for_center) / num_filtered_hist
    
    average_centers.append(center)
    average_heights.append(avg_height)

ax1.bar(average_centers, average_heights, width=0.8, 
        color= "#42bbd8", edgecolor='black', lw=0.5)

ax1.set_xlim(0, 30.5)
# Step 5: Use MaxNLocator to automatically adjust x-tick labels to show exactly 5 ticks
ax1.xaxis.set_major_locator(MaxNLocator(nbins=3))  # Automatically create 5 ticks
ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))









ax2 = fig.add_axes([0.25, 0.0, 0.15, 0.18])
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_title("Appropriation & circulation", fontsize=13)

# Step 1: Filter histograms whose systems belong to the "Resources input" group
filtered_hist = [
    (centers, heights, system) for centers, heights, system in hist if system in system_groups['Appropriation & circulation']
]

# Step 2: Merge centers and heights
merged_centers = []
merged_heights = []

for centers, heights, system in filtered_hist:
    merged_centers.extend(centers)
    merged_heights.extend(heights)

# Step 3: Find unique centers and calculate the average height for each center
unique_centers = sorted(set(merged_centers))

average_centers = []
average_heights = []

# Calculate average heights using the length of filtered_hist
num_filtered_hist = len(filtered_hist)

for center in unique_centers:
    # Find all heights corresponding to the current center
    heights_for_center = [height for c, height in zip(merged_centers, merged_heights) if c == center]
    
    # Sum all heights and divide by the total number of entries in filtered_hist
    avg_height = sum(heights_for_center) / num_filtered_hist
    
    average_centers.append(center)
    average_heights.append(avg_height)

ax2.bar(average_centers, average_heights, width=0.8, 
        color="#d7604d", edgecolor='black', lw=0.5)

ax2.set_xlim(0, 30.5)
# Step 5: Use MaxNLocator to automatically adjust x-tick labels to show exactly 5 ticks
ax2.xaxis.set_major_locator(MaxNLocator(nbins=3))  # Automatically create 5 ticks
ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))





ax3 = fig.add_axes([0.44, 0.0, 0.15, 0.18])
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_title("Transformation & conservation", fontsize=13)



# Step 1: Filter histograms whose systems belong to the "Resources input" group
filtered_hist = [
    (centers, heights, system) for centers, heights, system in hist if system in system_groups['Transformation & conservation']
]

# Step 2: Merge centers and heights
merged_centers = []
merged_heights = []

for centers, heights, system in filtered_hist:
    merged_centers.extend(centers)
    merged_heights.extend(heights)

# Step 3: Find unique centers and calculate the average height for each center
unique_centers = sorted(set(merged_centers))

average_centers = []
average_heights = []

# Calculate average heights using the length of filtered_hist
num_filtered_hist = len(filtered_hist)

for center in unique_centers:
    # Find all heights corresponding to the current center
    heights_for_center = [height for c, height in zip(merged_centers, merged_heights) if c == center]
    
    # Sum all heights and divide by the total number of entries in filtered_hist
    avg_height = sum(heights_for_center) / num_filtered_hist
    
    average_centers.append(center)
    average_heights.append(avg_height)

ax3.bar(average_centers, average_heights, width=0.8, 
        color="#17a388", edgecolor='black', lw=0.5)

ax3.set_xlim(0, 30.5)
# Step 5: Use MaxNLocator to automatically adjust x-tick labels to show exactly 5 ticks
ax3.xaxis.set_major_locator(MaxNLocator(nbins=3))  # Automatically create 5 ticks
ax3.yaxis.set_major_locator(MaxNLocator(nbins=3))




ax4 = fig.add_axes([0.63, 0.0, 0.15, 0.18])
ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_title("Consumption & excretion", fontsize=13)

# Step 1: Filter histograms whose systems belong to the "Resources input" group
filtered_hist = [
    (centers, heights, system) for centers, heights, system in hist if system in system_groups['Consumption & excretion']
]

# Step 2: Merge centers and heights
merged_centers = []
merged_heights = []

for centers, heights, system in filtered_hist:
    merged_centers.extend(centers)
    merged_heights.extend(heights)

# Step 3: Find unique centers and calculate the average height for each center
unique_centers = sorted(set(merged_centers))

average_centers = []
average_heights = []

# Calculate average heights using the length of filtered_hist
num_filtered_hist = len(filtered_hist)

for center in unique_centers:
    # Find all heights corresponding to the current center
    heights_for_center = [height for c, height in zip(merged_centers, merged_heights) if c == center]
    
    # Sum all heights and divide by the total number of entries in filtered_hist
    avg_height = sum(heights_for_center) / num_filtered_hist
    
    average_centers.append(center)
    average_heights.append(avg_height)

ax4.bar(average_centers, average_heights, width=0.8, 
        color= "#faa262", edgecolor='black', lw=0.5)

ax4.set_xlim(0, 30.5)
# Step 5: Use MaxNLocator to automatically adjust x-tick labels to show exactly 5 ticks
ax4.xaxis.set_major_locator(MaxNLocator(nbins=3))  # Automatically create 5 ticks
ax4.yaxis.set_major_locator(MaxNLocator(nbins=3))




ax5 = fig.add_axes([0.82, 0.0, 0.15, 0.18])
ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
ax5.tick_params(axis='both', which='major', labelsize=14)
ax5.set_title("Wastes output", fontsize=13)

# Step 1: Filter histograms whose systems belong to the "Resources input" group
filtered_hist = [
    (centers, heights, system) for centers, heights, system in hist if system in system_groups['Wastes output']
]

# Step 2: Merge centers and heights
merged_centers = []
merged_heights = []

for centers, heights, system in filtered_hist:
    merged_centers.extend(centers)
    merged_heights.extend(heights)

# Step 3: Find unique centers and calculate the average height for each center
unique_centers = sorted(set(merged_centers))

average_centers = []
average_heights = []

# Calculate average heights using the length of filtered_hist
num_filtered_hist = len(filtered_hist)

for center in unique_centers:
    # Find all heights corresponding to the current center
    heights_for_center = [height for c, height in zip(merged_centers, merged_heights) if c == center]
    
    # Sum all heights and divide by the total number of entries in filtered_hist
    avg_height = sum(heights_for_center) / num_filtered_hist
    
    average_centers.append(center)
    average_heights.append(avg_height)

ax5.bar(average_centers, average_heights, width=0.8, 
        color="#9c8ec1", edgecolor='black', lw=0.5)

ax5.set_xlim(0, 30.5)
# Step 5: Use MaxNLocator to automatically adjust x-tick labels to show exactly 5 ticks
ax5.xaxis.set_major_locator(MaxNLocator(nbins=3))  # Automatically create 5 ticks
ax5.yaxis.set_major_locator(MaxNLocator(nbins=3))




fig.text(0.11, 0.56, f"Total No. of nodes ($N$): {total_nodes:,.0f}", 
         fontsize=14, ha='left', va='top')
fig.text(0.11, 0.54, f"Total No. of edges ($E$): {total_edges:,.0f}", 
         fontsize=14, ha='left', va='top')
fig.text(0.11, 0.52, f"Total No. of hyperedges ($\mathcal{{E}}$): {total_hyperedges:,.0f}", 
         fontsize=14, ha='left', va='top')





# Convert to DataFrame
# Convert to DataFrame
df_scaling = pd.DataFrame(scaling)

# Remove rows where any value is 0
df_scaling = df_scaling[(df_scaling != 0).all(axis=1)]

df_scaling = np.log10(pd.DataFrame(df_scaling))

# Remove rows where any value is 0
df_scaling = df_scaling[(df_scaling != 0).all(axis=1)]







ax3 = fig.add_axes([0.405, 0.86, 0.085, 0.09])
# Scatter plot
ax3.scatter(df_scaling["Nodes"], df_scaling["Edges"], 
            color="blue", label="Data Points", alpha = 0.5)
# Remove x and y ticks and tick labels
ax3.set_xticks([])
ax3.set_yticks([])
ax3.tick_params(left=False, bottom=False)  # Removes the tick lines

# Add x and y axis titles
ax3.set_xlabel(r"$\log_{10}(N)$", fontsize=13)
ax3.set_ylabel(r"$\log_{10}(E)$", fontsize=13)

# Linear regression
nodes = np.array(df_scaling["Nodes"])
edges = np.array(df_scaling["Edges"])
m, b = np.polyfit(nodes, edges, 1)  # Fit a linear model: y = mx + b
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value1, std_err = stats.linregress(nodes, edges)

# Calculate R^2
predicted_edges = m * nodes + b
residuals = edges - predicted_edges
ss_res = np.sum(residuals**2)  # Residual sum of squares
ss_tot = np.sum((edges - np.mean(edges))**2)  # Total sum of squares
r_squared = 1 - (ss_res / ss_tot)

# Extend x range for the fit line
x_min = nodes.min() - 20  # Extend 20 units below the minimum
x_max = nodes.max() + 20  # Extend 20 units above the maximum
extended_x = np.linspace(x_min, x_max, 500)  # Generate extended x-values
extended_y = m * extended_x + b  # Calculate corresponding y-values

# Plot fit line
ax3.plot(extended_x, extended_y, color="red", )

# Display fit formula and R^2
ax3.text(0.5, 0.9, f"y = {m:.2f}x + {b:.1f}", 
         fontsize=11, ha='center', va='center', 
         transform=ax3.transAxes)
ax3.text(0.95, 0.1, f"$R^2$ = {r_squared:.2f}", 
         fontsize=11, ha='right', va='center', 
         transform=ax3.transAxes)
# Set axis limits to scatter data
ax3.set_xlim(nodes.min() - .5, nodes.max() + .5)  # Slight padding around scatter range
ax3.set_ylim(edges.min() - .5, edges.max() + .5)
ax3.text(0.95, .35, f"$p$=\n{p_value1:.1e}", 
         fontsize=11, ha='right', va='center', 
         transform=ax3.transAxes)







ax3 = fig.add_axes([0.405, 0.475, 0.085, 0.09])
# Scatter plot
ax3.scatter(df_scaling["Nodes"], df_scaling["Hyperedges"], 
            color="#0098d1", label="Data Points", alpha = 0.5)
# Remove x and y ticks and tick labels
ax3.set_xticks([])
ax3.set_yticks([])
ax3.tick_params(left=False, bottom=False)  # Removes the tick lines

# Add x and y axis titles
ax3.set_xlabel(r"$\log_{10}(N)$", fontsize=13)
ax3.set_ylabel(r"$\log_{10}(\mathcal{{E}})$", fontsize=13)

# Linear regression
nodes = np.array(df_scaling["Nodes"])
edges = np.array(df_scaling["Hyperedges"])
m, b = np.polyfit(nodes, edges, 1)  # Fit a linear model: y = mx + b
# Perform linear regression using scipy to get p-value of the slope
slope, intercept, r_value, p_value2, std_err = stats.linregress(nodes, edges)

# Calculate R^2
predicted_edges = m * nodes + b
residuals = edges - predicted_edges
ss_res = np.sum(residuals**2)  # Residual sum of squares
ss_tot = np.sum((edges - np.mean(edges))**2)  # Total sum of squares
r_squared = 1 - (ss_res / ss_tot)

# Extend x range for the fit line
x_min = nodes.min() - 20  # Extend 20 units below the minimum
x_max = nodes.max() + 20  # Extend 20 units above the maximum
extended_x = np.linspace(x_min, x_max, 500)  # Generate extended x-values
extended_y = m * extended_x + b  # Calculate corresponding y-values

# Plot fit line
ax3.plot(extended_x, extended_y, color="red", )

# Display fit formula and R^2
ax3.text(0.5, 0.9, f"y = {m:.2f}x {'+' if b >= 0 else '-'} {abs(b):.2f}", 
         fontsize=11, ha='center', va='center', 
         transform=ax3.transAxes)
ax3.text(0.95, 0.1, f"$R^2$ = {r_squared:.2f}", 
         fontsize=11, ha='right', va='center', 
         transform=ax3.transAxes)
# Set axis limits to scatter data
ax3.set_xlim(nodes.min() - 1.5, nodes.max() + 1.5)  # Slight padding around scatter range
ax3.set_ylim(edges.min() - 1.5, edges.max() + 1.5)
ax3.text(0.95, .35, f"$p$=\n{p_value2:.1e}", 
         fontsize=11, ha='right', va='center', 
         transform=ax3.transAxes)

# Save the plot as a PDF
pdf_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Fig5.pdf"
plt.savefig(pdf_file, format="pdf", bbox_inches="tight")  # Save as PDF with tight layout

# Show the plot
plt.tight_layout()



# Show the figure
plt.show()


















# Determine the length of the lists (assuming all lists are the same length)
num_rows = len(next(iter(group1.values())))

# Prepare the data for writing to CSV
header = list(group1.keys())  # Column names
rows = zip(*group1.values())  # Transpose the list of values to get each row

# Write to CSV
with open('/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/3rd.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header
    writer.writerows(rows)   # Write the rows





# Determine the length of the lists (assuming all lists are the same length)
num_rows = len(next(iter(group2.values())))

# Prepare the data for writing to CSV
header = list(group2.keys())  # Column names
rows = zip(*group2.values())  # Transpose the list of values to get each row

# Write to CSV
with open('/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/4th.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the header
    writer.writerows(rows)   # Write the rows












