import matplotlib.pyplot as plt
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

import xgi


fig, ax = plt.subplots(figsize=(15, 13), dpi = 600)
# Remove all elements from the plot
ax.axis('off')  # Turn off the axes

# Set a background color if desired (optional)
fig.patch.set_facecolor('white')  # Change 'white' to any color you prefer












# Paths to the PNG images
p11 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/11.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p11)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.28, 0.87), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p12 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/12.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p12)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.49, 0.87), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p13 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/13.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p13)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.7, 0.87), frameon=False)
ax.add_artist(ab_hf)






# Paths to the PNG images
p14 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/14.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p14)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.91, 0.87), frameon=False)
ax.add_artist(ab_hf)











# Paths to the PNG images
p21 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/21.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p21)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.28, 0.71), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p22 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/22.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p22)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.49, 0.67), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p23 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/23.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p23)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.7, 0.67), frameon=False)
ax.add_artist(ab_hf)






# Paths to the PNG images
p24 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/24.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p24)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.91, 0.67), frameon=False)
ax.add_artist(ab_hf)








# Paths to the PNG images
p31 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/31.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p31)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.28, 0.51), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p32 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/32.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p32)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.49, 0.47), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p33 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/33.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p33)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.7, 0.47), frameon=False)
ax.add_artist(ab_hf)






# Paths to the PNG images
p34 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/34.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p34)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.91, 0.47), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p41 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/41.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p41)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.28, 0.33), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p42 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/42.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p42)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.49, 0.27), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p43 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/43.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p43)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.7, 0.27), frameon=False)
ax.add_artist(ab_hf)






# Paths to the PNG images
p44 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/44.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p44)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.91, 0.27), frameon=False)
ax.add_artist(ab_hf)











# Paths to the PNG images
p51 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/51.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p51)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.28, 0.12), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p52 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/52.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p52)

imagebox_hf = OffsetImage(hf_img, zoom=0.27)
ab_hf = AnnotationBbox(imagebox_hf, (0.49, 0.05), frameon=False)
ax.add_artist(ab_hf)









# Paths to the PNG images
p53 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/53.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p53)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.7, 0.05), frameon=False)
ax.add_artist(ab_hf)






# Paths to the PNG images
p54 = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/③Metamodel_training/54.png"  # Path to Hugging Face logo

# Load the images
hf_img = mpimg.imread(p54)

imagebox_hf = OffsetImage(hf_img, zoom=0.3)
ab_hf = AnnotationBbox(imagebox_hf, (0.91, 0.05), frameon=False)
ax.add_artist(ab_hf)
















# Add annotations
ax.annotate("Resources\ninput:", xy=(0.001, 0.9), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Fishing,\nwind energy,\nwave energy\nharvesting and\ncollecting", xy=(0.001, 0.795),
            fontsize=13, ha='left')

ax.annotate("Appropriation\nand\ncirculation:", xy=(0.001, 0.68), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Port operations,\nshipbuilding,\nfreight shipping", xy=(0.001, 0.615),
            fontsize=13, ha='left')

ax.annotate("Transformation\nand\nconservation:", xy=(0.001, 0.49), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Marine protected\nareas,coastal\ninfrastructure,\naquaculture\ndesalination", xy=(0.001, 0.385),
            fontsize=13, ha='left')

ax.annotate("Consumption\nand\nexcretion:", xy=(0.001, 0.3), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Ship emissions,\nhydrocarbon\nextraction,\nland reclamation", xy=(0.001, 0.215),
            fontsize=13, ha='left')

ax.annotate("Wastes\noutput:", xy=(0.001, 0.12), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Carbon sequestration,\nland-based effluent,\nwastewater treatment", xy=(0.001, 0.055),
            fontsize=13, ha='left')



ax.annotate("Hypergraph metamodel\nevolution rule", xy=(0.28, 0.97), weight='bold',
            fontsize=13, ha='center')
ax.annotate("Evolution step 1", xy=(0.49, 0.99), weight='bold',
            fontsize=13, ha='center')
ax.annotate("Evolution step 3", xy=(0.7, 0.99), weight='bold',
            fontsize=13, ha='center')
ax.annotate("Evolution step 5", xy=(0.91, 0.99), weight='bold',
            fontsize=13, ha='center')







ax.annotate("{{1, 2, 3}} ->\n{{1, 2, 3}, {4, 5, 6},\n{1, 4, 6}, {3, 7, 5}}", 
            xy=(0.28, 0.763), 
            fontsize=13, ha='center')
ax.annotate("{{1, 2, 3}} ->\n{{1, 2, 3}, {4, 5, 7},\n{4, 5, 6}, {4, 5, 6, 7},\n{1, 4, 6}, {3, 7, 5}}", 
            xy=(0.28, 0.605), 
            fontsize=13, ha='center')
ax.annotate("{{1, 2, 3}} ->\n{{1, 2, 3}, {4, 6, 7},\n{4, 5, 6}, {1, 4, 6},\n{3, 7, 5}}, {{0, 0, 0}}", 
            xy=(0.28, 0.405), 
            fontsize=13, ha='center')
ax.annotate("{{1, 2, 3}} ->\n{{1, 2, 3}, {5, 6, 7},\n{4, 5, 6}, {4, 6, 7},\n{1, 4, 6}, {3, 7, 5}}", 
            xy=(0.28, 0.225), 
            fontsize=13, ha='center')
ax.annotate("{{1, 2, 3}} ->\n{{1, 2, 3}, {4, 6, 7},\n{4, 5, 7}, {4, 5, 6},\n{4, 5, 6, 7}, {1, 4, 6}, {3, 7, 5}}", 
            xy=(0.28, 0.02), 
            fontsize=13, ha='center')









# Save the plot as a PDF
pdf_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Fig7.pdf"
plt.savefig(pdf_file, format="pdf", bbox_inches="tight")  # Save as PDF with tight layout

# Show the plot
plt.tight_layout()

plt.show()














