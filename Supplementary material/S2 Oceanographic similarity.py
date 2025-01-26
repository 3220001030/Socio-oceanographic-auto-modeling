import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Define a set of words to compare
words = ["Global",
         "Climate",
         "Ecosystem",
         "Coast",
         "Mangrove",
         
         "Region",
         "Weather",  
         "Sea",
         "Water",
         "Landscape",
         
         "Local",
         "Connectivity",
         "Community",
         "Fish",
         "Temperature",
         
         "Seaweed",
         "Planet",
         "Waves",
         "Gulf",
         "Seawater",
         
         
         
         "Environment",
         "Shoal",
         "Delta",
         "Dolphin",
         "Shore",]

reference_word = "Global"

# Create spaCy tokens for each word
tokens = [nlp(word) for word in words]

# Initialize a similarity matrix
similarity_matrix = np.zeros((len(words), len(words)))

# Compute similarity scores
for i, token1 in enumerate(tokens):
    for j, token2 in enumerate(tokens):
        if i != j:
            similarity_matrix[i, j] = token1.similarity(token2)

# Convert the matrix into a DataFrame for better visualization
df_similarity = pd.DataFrame(similarity_matrix, index=words, columns=words)

similarities_with_reference = df_similarity[reference_word].drop(index=reference_word).sort_values(ascending=True)

# Append the reference word at the end of the sorted list
new_order = list(similarities_with_reference.index) + [reference_word]

# Step 2: Reorder both rows and columns based on this order
df_reordered = df_similarity.loc[new_order, new_order]


# Plot the similarity matrix using seaborn
plt.figure(figsize=(14, 12), dpi = 300)

mask = np.triu(np.ones_like(df_similarity, dtype=bool))
sns.heatmap(df_reordered, mask = mask,
            annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
# Get the current axis
ax = plt.gca()

# Get the x-tick positions and labels
xticks = ax.get_xticks()
xtick_labels = [label.get_text() for label in ax.get_xticklabels()]

# Define custom annotations and their positions
annotations = ["Marine\n species",
               "Aquatic\n environment",
               "Macro\n conditions",
               "Regional\n seas",
               "Global\noceans"]
annotation_positions = [np.mean(xticks[i:i + 5]) for i in range(0, len(xticks), 5)]

# Draw brackets and add annotations
for start, pos, text in zip(range(0, len(xticks), 5), annotation_positions, annotations):
    # Calculate bracket start and end points
    left = xticks[start]  # Start of the bracket
    right = xticks[start + 4] if start + 4 < len(xticks) else xticks[-1]  # End of the bracket
    mid = (left + right) / 2  # Center for the annotation

    # Draw bracket using a curve or line
    ax.plot([left, left, right, right],  # X coordinates
            [-0.12, -0.13, -0.13, -0.12],  # Y coordinates
            color='blue', linewidth=2, transform=ax.get_xaxis_transform(), clip_on=False)

    # Add annotation below the bracket
    ax.text(mid, -0.14, text, ha='center', va='top', fontsize=14, color="black", transform=ax.get_xaxis_transform())

# Customize font size for tick labels
ax.tick_params(axis='both', labelsize=12)

# Adjust plot layout
plt.tight_layout()
plt.show()


