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
from matplotlib.ticker import MaxNLocator
dimensions = pd.read_csv("/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/webofscience/type_and_slope_output.csv")

# Define the directory where your CSV and XLSX files are stored
directory = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/矩阵数据/"  # Replace with your actual directory path

# Function to load CSV files
def load_csv(file_path):
    df = pd.read_csv(file_path, header=None)  # Assume the first column is the row index
    df = df.fillna(0)
    system = df.iloc[0, 0]

    
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


# Example of using the function
matrices = load_data_from_directory(directory)


# Display the loaded matrices
for file_path, matrix_info in matrices.items():
    adj_matrix = matrix_info["matrix"]
    system = matrix_info["system"]
    
    
    
    # Step 1: Truncate the DataFrame to ensure length = width (making it square)
    size = min(adj_matrix.shape[0], adj_matrix.shape[1])
    adj_matrix = adj_matrix.iloc[:size, :size]
    
    # Step 2: Fill all NA with 0
    adj_matrix = adj_matrix.fillna(0)
    
    # Step 3: Remove the first cell (top-left)
    adj_matrix.iloc[0, 0] = ""
    
    # Step 4: Convert the DataFrame to a string representation
    string_representation = adj_matrix.to_csv(index=False, header=False)
    
    # Output the result
    print(string_representation)
    
    
    txt_file_path = file_path.replace(".csv", ".txt")
    
    
    with open(txt_file_path, "w") as file:
        file.write(string_representation)
        
    print(f"String representation written to {txt_file_path}")











