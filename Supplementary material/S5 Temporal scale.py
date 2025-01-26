import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Define a set of words to compare
words = ["Hour",
         "Daily",
         "Weekly",
         "Monthly",
         "Yearly"]

reference_word = "Yearly"

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
plt.figure(figsize=(5, 5), dpi = 300)

mask = np.triu(np.ones_like(df_similarity, dtype=bool))
sns.heatmap(df_reordered, mask = mask,
            annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
# Get the current axis
ax = plt.gca()

# Get the x-tick positions and labels
xticks = ax.get_xticks()
xtick_labels = [label.get_text() for label in ax.get_xticklabels()]



# Customize font size for tick labels
ax.tick_params(axis='both', labelsize=12)

# Adjust plot layout
plt.tight_layout()
plt.show()


