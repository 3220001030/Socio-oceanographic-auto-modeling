import pandas as pd
import numpy as np
import networkx as nx
import csv

# Load the CSV file
csv_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Validation.csv"  # Replace with your actual file path
df = pd.read_csv(csv_file, dtype=str)  # Read CSV as string to avoid parsing errors

# Function to extract a matrix from the custom CSV format, ignoring text before the first comma
def extract_matrix(cell_content):
    lines = cell_content.strip().split("\n")  # Split content by newlines
    matrix_data = []

    for line in lines[1:]:  # Skip the first header line
        # Extract part after the first comma
        after_comma = line.split(",", 1)[-1]  # Take everything after the first comma
        entries = after_comma.split(",")  # Split remaining part into elements

        # Clean extra spaces and remove quotes if present
        cleaned_entries = [entry.strip().replace('"', '') for entry in entries]

        try:
            row = list(map(int, cleaned_entries))  # Convert to integers
            matrix_data.append(row)
        except ValueError:
            print(f"Skipping invalid row: {line}")
            continue  # Skip problematic rows

    return np.array(matrix_data)

# Dictionary to store accumulated costs for each stage
edit_path_results = {}

# Process each unique Stage in the CSV
for stage in df['Stage'].unique():
    stage_data = df[df['Stage'] == stage]

    # Initialize cumulative values for cost calculation
    total_cost1_list = []
    total_cost2_list = []
    total_cost3_list = []

 

    try:
        for _, row in stage_data.iterrows():
            
            A0 = extract_matrix(row['Original'])
            A1 = extract_matrix(row['Phi 4'])
            A2 = extract_matrix(row['Fine-tuned'])
            A3 = extract_matrix(row['GPT4o'])

            # Convert matrices to NetworkX graphs
            G0 = nx.from_numpy_array(A0)
            G1 = nx.from_numpy_array(A1)
            G2 = nx.from_numpy_array(A2)
            G3 = nx.from_numpy_array(A3)

            # Compute optimal edit paths
            _, cost1 = nx.optimal_edit_paths(G0, G1)
            _, cost2 = nx.optimal_edit_paths(G0, G2)
            _, cost3 = nx.optimal_edit_paths(G0, G3)
            

            # Append costs to lists for each stage
            total_cost1_list.append(cost1)
            total_cost2_list.append(cost2)
            total_cost3_list.append(cost3)  # You can modify this depending on your requirement


        # Store results in dictionary
        if stage not in edit_path_results:
            edit_path_results[stage] = {"total_cost1": 0, 
                                        "total_cost2": 0,
                                        "total_cost3": 0}

        # Calculate avg and std for the costs
        avg_cost1 = np.mean(total_cost1_list)
        std_cost1 = np.std(total_cost1_list)
        avg_cost2 = np.mean(total_cost2_list)
        std_cost2 = np.std(total_cost2_list)
        avg_cost3 = np.mean(total_cost3_list)
        std_cost3 = np.std(total_cost3_list)


        # Store results in dictionary
        edit_path_results[stage] = {
            "avg_cost1": avg_cost1,
            "std_cost1": std_cost1,
            "avg_cost2": avg_cost2,
            "std_cost2": std_cost2,
            "avg_cost3": avg_cost3,
            "std_cost3": std_cost3,

        }


        print(f"Stage: {stage}")
        print("Avg cost1:", avg_cost1)
        print("Avg cost2:", avg_cost2)
        print("Avg cost3:", avg_cost3)
        print("-" * 40)

    except Exception as e:
        print(f"Error processing stage {stage}: {A0}")






# Save the accumulated edit path results to a CSV file
output_csv_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Comparison.csv"

# Prepare the data for CSV
csv_data = []
for stage, results in edit_path_results.items():
    csv_data.append({
        "Stage": stage,
        "Avg Cost 1": results["avg_cost1"],
        "Avg Cost 2": results["avg_cost2"],
        "Avg Cost 3": results["avg_cost3"],
        "Std Cost 1": results["std_cost1"],
        "Std Cost 2": results["std_cost2"],
        "Std Cost 3": results["std_cost3"],

    })

# Write to CSV
with open(output_csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["Stage", 
                                           "Avg Cost 1", 
                                           "Avg Cost 2",
                                           "Avg Cost 3",
                                           "Std Cost 1",
                                           "Std Cost 2",
                                           "Std Cost 3",
                                           ])
    writer.writeheader()
    writer.writerows(csv_data)
    
    
    
    