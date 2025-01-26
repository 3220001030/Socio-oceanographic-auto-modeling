import matplotlib.pyplot as plt
import inspect
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms
import xgi
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import pandas as pd


b = "#99d5ef"
g = "#26aa92"
r = "#ffd3c8"
y = "#e3c800"
p = "purple"
node = False


def add_fancy_patch_around(ax, bb, fc, ec, **kwargs):
    kwargs = {
        'facecolor': fc,
        'edgecolor': ec,
        **kwargs
    }
    fancy = FancyBboxPatch(bb.p0, bb.width, bb.height, **kwargs)
    ax.add_patch(fancy)
    return fancy




fig = plt.figure(figsize=(12, 8), dpi = 600)
grid = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])

# Adjust layout: Increase vertical space between rows
plt.subplots_adjust(hspace=0.15)  # You can adjust this value (e.g., 0.5) to get more space


ax1 = fig.add_subplot(grid[0, 0])
ax2 = fig.add_subplot(grid[0, 1])
ax3 = fig.add_subplot(grid[0, 2])

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')















bb11 = mtransforms.Bbox([[0.1, 0.42], 
                       [0.9, 0.8]])

# a fancy box with round corners. pad=0.1
add_fancy_patch_around(ax1, bb11, 
                       "#effdfe", "#42bbd8",
                       boxstyle="round,pad=0.05")


ax1.text(.07, .97, "1.Dataset preparation", 
        color="black",weight="bold",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)

ax1.text(.07, .9, "{“conversations”: [{“role”: “system”, \n“content”:", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)

ax1.text(.07, .61, "“You are a socio-oceanography\nauto-modeling bot whose primary\ngoal is to help users build models\nin adjacency matrix forms based\non their descriptions. You are\nfriendly and concise. You only\nprovide robust answers to queries\nand do not provide answers that\nare not based on higher-order\noceanographic or social\nhypergraphs.”", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)


ax1.text(.07, .32, "},{“role”: “user”, “content”:", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)


bb12 = mtransforms.Bbox([[0.1, 0.055], 
                       [0.9, 0.23]])

# a fancy box with round corners. pad=0.1
add_fancy_patch_around(ax1, bb12, 
                       "#fff1e8", "#faa262",
                       boxstyle="round,pad=0.05")



ax1.text(.07, .14, "“Marine protected areas are\ncurrently recognized as an\nalternative for the conservation of\nmarine ecosystems. Although the\nprotection reduces the area\navailable for fishing...”", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)


ax_arrow = fig.add_axes([0.33, 0.58, 0.1, 0.06])
ax_arrow.axis('off')
arrow = mpatches.Arrow(0.2, 0.5, 0.5, 0,
                       color = "#faa262")
ax_arrow.add_patch(arrow)






bb21 = mtransforms.Bbox([[0.1, 0.5], 
                       [0.9, 0.945]])

# a fancy box with round corners. pad=0.1
add_fancy_patch_around(ax2, bb21, 
                       "#fff1e8", "#faa262",
                       boxstyle="round,pad=0.05")



ax2.text(.07, .89, "Create an adjacency matrix with\nreference but not limited to these\nsocial and oceanographic \nhypergraphs:", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)




ax_hypergraph = fig.add_axes([0.41, 0.63, 0.08, 0.08])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[1, 2], [1, 2, 3], [1, 3]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.3,node_labels=node,
    pos=pos,
    edge_fc=[b],
    node_size=5,dyad_lw=2)
ax2.annotate("Species\ndynamics", xy=(0.2, 0.705), 
            fontsize=9, ha='center')


ax_hypergraph = fig.add_axes([0.4725, 0.63, 0.08, 0.08])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 6, 7], [5, 7, 8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.3,node_labels=node,
    pos=pos,
    edge_fc=[g,y],
    node_size=5,dyad_lw=2)
ax2.annotate("Social-\necological\nfit", xy=(0.8, 0.67), 
            fontsize=9, ha='center')



ax_hypergraph = fig.add_axes([0.54, 0.63, 0.08, 0.08])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 6, 7, 8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.3,node_labels=node,
    pos=pos,
    edge_fc=[r],
    node_size=5,dyad_lw=2)
ax2.annotate("Social\nmanagement", xy=(0.5, 0.705), 
            fontsize=9, ha='center')






ax_arrow = fig.add_axes([0.605, 0.58, 0.1, 0.06])
ax_arrow.axis('off')
arrow = mpatches.Arrow(0.2, 0.5, 0.5, 0,
                       color = "#17a388")
ax_arrow.add_patch(arrow)








ax2.text(.07, .4, "},{“role”: “assistant”, “content”:", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)











bb22 = mtransforms.Bbox([[0.1, 0.055], 
                       [0.9, 0.3]])

# a fancy box with round corners. pad=0.1
add_fancy_patch_around(ax2, bb22, 
                       "#dafff5", "#17a388",
                       boxstyle="round,pad=0.05")






n = 6
# Create a binary matrix
matrix = np.random.choice([0, 1], size=(n, n), p=[0.5, 0.5])

# Get the dimensions of the matrix
rows, cols = matrix.shape

# Create a meshgrid for the coordinates
x = np.arange(cols + 1)
y = np.arange(rows + 1)
ax_matrix = fig.add_axes([0.42, 0.42, 0.16, 0.135]) 
# Use pcolormesh to draw the matrix

ax_matrix.pcolormesh(x, y, 
               matrix, cmap=ListedColormap(["#dafff5", "#17a388"]), 
               edgecolors="black", linewidth=2)

# Remove axis ticks and labels
ax_matrix.set_xticks([])
ax_matrix.set_yticks([])
ax_matrix.axis('off')
# Keep the origin at the top-left (matrix convention)
ax_matrix.invert_yaxis()









ax2.text(.84, .05, "}]}", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)

azure_ai_image = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/MS.png"  # Path to Azure AI content image

azure_img = mpimg.imread(azure_ai_image)
# Add MS Azure AI content image to the subplot
imagebox_azure = OffsetImage(azure_img, zoom=0.07)


ab_azure = AnnotationBbox(imagebox_azure, (0.5, 0.37), frameon=False)
ax3.add_artist(ab_azure)




bb31 = mtransforms.Bbox([[0.1, 0.67], 
                       [0.9, 0.88]])

# a fancy box with round corners. pad=0.1
add_fancy_patch_around(ax3, bb31, 
                       "white", "gray",
                       boxstyle="round,pad=0.05")


bb32 = mtransforms.Bbox([[0.1, 0.3], 
                       [0.9, 0.53]])

# a fancy box with round corners. pad=0.1
add_fancy_patch_around(ax3, bb32, 
                       "white", "gray",
                       boxstyle="round,pad=0.05")






bb33 = mtransforms.Bbox([[0.1, 0.055], 
                       [0.9, 0.16]])

# a fancy box with round corners. pad=0.1
add_fancy_patch_around(ax3, bb33, 
                       "white", "gray",
                       boxstyle="round,pad=0.05")






# Load the images
colab_img = mpimg.imread("/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Colab.png")
unsloth_img = mpimg.imread("/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Unsloth.png")




# Add Hugging Face logo to the subplot
imagebox_hf = OffsetImage(colab_img, zoom=0.045)
ab_hf = AnnotationBbox(imagebox_hf, (0.75, 0.72), frameon=False)
ax3.add_artist(ab_hf)



# Add MS Azure AI content image to the subplot
imagebox_azure = OffsetImage(unsloth_img, zoom=0.1)
ab_azure = AnnotationBbox(imagebox_azure, (0.35, 0.72), frameon=False)
ax3.add_artist(ab_azure)




ax3.text(.3, .87, "Lightweight fine-\ntuning library", 
        color="black",
        horizontalalignment="center", verticalalignment="center",
        wrap = True)


ax3.text(.74, .85, "Online hosted\ncomputer\nresources", 
        color="black",
        horizontalalignment="center", verticalalignment="center",
        wrap = True)










# Paths to the PNG images
hugging_face_logo = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/HF.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(hugging_face_logo)




# Add Hugging Face logo to the subplot


ax3.text(.5, .17, "Open-source platform", 
        color="black",
        horizontalalignment="center", verticalalignment="center",
        wrap = True)





ax3.text(.5, .52, "Academically trained LLM\nwith superb logical deduction", 
        color="black",
        horizontalalignment="center", verticalalignment="center",
        wrap = True)

imagebox_hf = OffsetImage(hf_img, zoom=0.045)
ab_hf = AnnotationBbox(imagebox_hf, (0.5, 0.08), frameon=False)
ax3.add_artist(ab_hf)






ax3.text(.07, .97, "2.Fine-tuning", 
        color="black",weight="bold",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)













ax4 = fig.add_subplot(grid[1, 0])



# Load the CSV file
csv_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Loss.csv"  # Replace with your actual file path
df = pd.read_csv(csv_file)

# Calculate the moving average (e.g., window size of 10)
window_size = 10
df['Moving Average'] = df['Training loss'].rolling(window=window_size).mean()


# Plot training loss
ax4.plot(df['Step'], df['Training loss'], label='Training Loss', color='blue', linewidth=1)

# Plot moving average
ax4.plot(df['Step'], df['Moving Average'], label=f'{window_size}-Step Moving Average', color='red', linewidth=2)

# Customize the plot

ax4.set_xlabel("Step")

ax4.legend(frameon = False)


ax4.text(.07, .12, "Learning_rate: 2e-4\nTraining time: 3926.33s\nPeak reserved memory: 13.396GB", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)


ax4.text(90, .47, "GPU: NVIDIA A100 Tensor Core\n(83.5 GB RAM)", 
        color="black",
        horizontalalignment="right", verticalalignment="center",
        wrap = True)








ax5 = fig.add_subplot(grid[1, 1:])




ax1.set_title(r"$\mathbf{{A}}$ Workflow of pre-training and fine-tuning", 
                  fontsize=12, loc='left')
ax4.set_title(r"$\mathbf{{B}}$ Training loss", 
              fontsize=12, loc='left')
ax5.set_title(r"$\mathbf{{C}}$ Optimal edit path costs", 
              fontsize=12, loc='left')






# Load the Comparison.csv file
comparison_csv = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Comparison.csv"
comparison_data = pd.read_csv(comparison_csv)

# Extract data for plotting
stages = comparison_data["Stage"]
total_cost1 = comparison_data["Avg Cost 1"]
total_cost2 = comparison_data["Avg Cost 2"]
total_cost3 = comparison_data["Avg Cost 3"]

std_cost1 = comparison_data["Std Cost 1"]
std_cost2 = comparison_data["Std Cost 2"]
std_cost3 = comparison_data["Std Cost 3"]

# Plot bar chart
bar_width = 0.25
x = range(len(stages))

ax5.bar([i - bar_width for i in x], 
        total_cost3, bar_width, yerr = std_cost3,
        label="GPT 4o without hypergraph prompts", color="#d7604d")
ax5.bar([i  for i in x], 
        total_cost1, bar_width, yerr = std_cost1,
        label="Phi-4 with hypergraph prompts", color="#42bbd8")
ax5.bar([i + bar_width for i in x], 
        total_cost2, bar_width, yerr = std_cost2,
        label="Phi-4 with hypergraph fine-tuning", color="#faa262")
ax5.set_ylim(0, 20)
# Customize the chart
ax5.set_xticks(x)

ax5.set_xlabel("Sociometabolic stage")
ax5.legend(loc='upper left', bbox_to_anchor=(.08, 1),
           frameon = False)  # Move the legend to the right

ax5.set_xticklabels(["Resources\ninput",
                     "Appropriation &\ncirculation",
                     "Transformation &\nconservation",
                     "Consumption &\nexcretion",
                     "Wastes\noutput"], ha="center",
                    fontsize = 9)  # Apply custom tick labels


ax5.text(.24, 11.8, f"$n$=50", 
        color="black",
        horizontalalignment="left", verticalalignment="center",
        wrap = True)


# Save the plot as a PDF
pdf_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Fig8.pdf"
plt.savefig(pdf_file, format="pdf", bbox_inches="tight")  # Save as PDF with tight layout

# Show the plot
plt.tight_layout()



plt.show()








