import matplotlib.pyplot as plt

import xgi


fig, ax = plt.subplots(figsize=(15, 15), dpi = 600)
# Remove all elements from the plot
ax.axis('off')  # Turn off the axes

# Set a background color if desired (optional)
fig.patch.set_facecolor('white')  # Change 'white' to any color you prefer


b = "#99d5ef"
g = "#26aa92"
r = "#ffd3c8"
y = "#e3c800"
p = "purple"

node = True


# Line 1
ax_hypergraph = fig.add_axes([0.3, 0.7, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 3], [2, 3],
     ])

pos = xgi.circular_layout(H)
xgi.draw(H,node_labels=node,
    pos=pos,
    edge_fc=[b],
    node_size=15,dyad_lw=3)
ax.annotate("(0.276)", xy=(0.37, 0.775), 
            fontsize=13, ha='center')
ax.annotate("Species/\nturbines", xy=(0.245, 0.91), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Environment", xy=(0.245, 0.79), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Climate", xy=(0.377, 0.835), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Species/\nhydro/aero-\ndynamics", xy=(0.305, 0.842), 
            fontsize=13, ha='center')

ax_hypergraph = fig.add_axes([0.5, 0.7, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [
     #4th
     [4, 5], [4, 5, 6], [4, 6], [4, 7], [5, 6], [5, 7]])

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[g],
    node_size=15,dyad_lw=3)
ax.annotate("(0.112)", xy=(0.64, 0.775), 
            fontsize=13, ha='center')
ax.annotate("Fisher/operator", xy=(0.568, 0.927), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Governance", xy=(0.568, 0.79), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Fuel", xy=(0.47, 0.835), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Market", xy=(0.655, 0.835), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Supply-\ndemand", xy=(0.67, 0.91), 
            fontsize=12, ha='center')

ax_hypergraph = fig.add_axes([0.7, 0.7, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 3], [2, 3],
     #4th
     [4, 5], [4, 5, 6], [4, 6], [4, 7], [5, 6], [5, 7],
    #inter
    [1,4], [3,7]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[g],
    node_size=15,dyad_lw=3)




ax_hypergraph = fig.add_axes([0.85, 0.785, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[1,2],[1,4],[2,3],[3,4]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[g],
    node_size=5,dyad_lw=2)
ax.annotate("High\nconnectivity", xy=(0.96, 0.93), 
            fontsize=12, ha='center')



ax_hypergraph = fig.add_axes([0.85, 0.708, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 6, 7]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[g],
    node_size=5,dyad_lw=2)
ax.annotate("Resources\nmanagement", xy=(0.96, 0.83), 
            fontsize=12, ha='center')












# Line 2
ax_hypergraph = fig.add_axes([0.3, 0.55, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 2, 3], [1, 3],
     ],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[b],
    node_size=15,dyad_lw=3)
ax.annotate("(0.274)", xy=(0.37, 0.58), 
            fontsize=13, ha='center')
ax.annotate("Waves", xy=(0.245, 0.735), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Weather", xy=(0.245, 0.593), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Climate", xy=(0.377, 0.64), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Hydro-\ndynamics", xy=(0.305, 0.66), 
            fontsize=13, ha='center',zorder=30000)


# Redraw the figure with all elements, forcing the annotation to be redrawn on top
fig.canvas.flush_events()
ax_hypergraph = fig.add_axes([0.5, 0.55, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#4th
     [4, 5, 6], [4, 5, 6, 7], [4, 5, 7],
     [4, 5]
    ],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[g, r, y],
    node_size=15,dyad_lw=3)
ax.annotate("(0.093)", xy=(0.64, 0.58), 
            fontsize=13, ha='center')
ax.annotate("Ship", xy=(0.568, 0.735), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Governance", xy=(0.568, 0.593), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Traveler", xy=(0.47, 0.64), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Port", xy=(0.655, 0.64), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Gravity model\nof migration", xy=(0.67, 0.73), 
            fontsize=12, ha='center')

ax_hypergraph = fig.add_axes([0.7, 0.55, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 2, 3], [1, 3],
     #4th
     [4, 5, 6], [4, 5, 6, 7], [4, 5, 7],
     [4, 5],
    #inter
    [1,4], [3,7]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[b,g,r,y],
    node_size=15,dyad_lw=3)


ax_hypergraph = fig.add_axes([0.85, 0.63, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 6, 7, 8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[r],
    node_size=5,dyad_lw=2)
ax.annotate("Social\nmanagement", xy=(0.96, 0.73), 
            fontsize=12, ha='center')


ax_hypergraph = fig.add_axes([0.85, 0.555, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 6, 8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[y],
    node_size=5,dyad_lw=2)
ax.annotate("Infrastructure\nmanagement", xy=(0.96, 0.63), 
            fontsize=12, ha='center')






# Line 3
ax_hypergraph = fig.add_axes([0.3, 0.4, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 2, 3], [1, 3]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[b],
    node_size=15,dyad_lw=3)
ax.annotate("(0.228)", xy=(0.37, 0.385), 
            fontsize=13, ha='center')
ax.annotate("Species/\nresources", xy=(0.245, 0.522), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Environment", xy=(0.245, 0.4), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Climate", xy=(0.377, 0.445), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Species/\nresources\ngrowth", xy=(0.305, 0.45), 
            fontsize=13, ha='center')

ax_hypergraph = fig.add_axes([0.5, 0.4, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#4th
     [4, 5], [4, 5, 6], [4, 5, 6, 7], [4, 6, 7]
     ],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[g,r,y,],
    node_size=15,dyad_lw=3)
ax.annotate("(0.100)", xy=(0.64, 0.385), 
            fontsize=13, ha='center')
ax.annotate("Citizen", xy=(0.568, 0.537), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Designer", xy=(0.568, 0.4), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Maintenance", xy=(0.47, 0.445), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Admin", xy=(0.655, 0.445), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Agent-based\ninteractions", xy=(0.67, 0.53), 
            fontsize=12, ha='center')

ax_hypergraph = fig.add_axes([0.7, 0.4, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 2, 3], [1, 3],
     #4th
     [5, 6], [5, 6, 7], [5, 6, 7, 8], [5, 7, 8],
    #inter
    [1,5], [3,8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[b,g,r, y],
    node_size=15,dyad_lw=3)

ax_hypergraph = fig.add_axes([0.85, 0.485, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 6, 7], [5, 7, 8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[g,y],
    node_size=5,dyad_lw=2)
ax.annotate("Social-ecological\nfit", xy=(0.96, 0.54), 
            fontsize=12, ha='center')

ax_hypergraph = fig.add_axes([0.85, 0.405, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[1,2],[1,4],[2,3]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[],
    node_size=5,dyad_lw=2)
ax.annotate("Middle \nconnectivity", xy=(0.96, 0.435), 
            fontsize=12, ha='center')






# Line 4
ax_hypergraph = fig.add_axes([0.3, 0.25, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 2, 3], [1, 3],
     ],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[b],
    node_size=15,dyad_lw=3)
ax.annotate("(0.207)", xy=(0.37, 0.19), 
            fontsize=13, ha='center')
ax.annotate("Species", xy=(0.245, 0.345), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Environment", xy=(0.245, 0.205), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Effluent", xy=(0.377, 0.25), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Effluent/\nsediment\ndiffusion", xy=(0.305, 0.255), 
            fontsize=13, ha='center')

ax_hypergraph = fig.add_axes([0.5, 0.25, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#4th
     [4, 5], [4, 6, 7],
     [4, 5], [4, 5, 6], [4, 6], [5, 6, 7]
    ],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[g,r,y],
    node_size=15,dyad_lw=3)
ax.annotate("(0.218)", xy=(0.64, 0.19), 
            fontsize=13, ha='center')
ax.annotate("People", xy=(0.568, 0.345), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Governance", xy=(0.568, 0.205), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Factory", xy=(0.47, 0.25), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Machine", xy=(0.655, 0.25), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Production\nfunction", xy=(0.67, 0.33), 
            fontsize=12, ha='center')

ax_hypergraph = fig.add_axes([0.7, 0.25, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 2, 3], [1, 3],
     #4th
     [4, 5], [4, 6, 7],
     [4, 5], [4, 5, 6], [4, 6], [5, 6, 7],
    #inter
    [1,4], [3,7]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[b,g,r,y],
    node_size=15,dyad_lw=3)


ax_hypergraph = fig.add_axes([0.85, 0.33, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 7, 8], [5, 6, 7]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[g,r],
    node_size=5,dyad_lw=2)
ax.annotate("Socio-economic\noverlap", xy=(0.96, 0.34), 
            fontsize=12, ha='center')


ax_hypergraph = fig.add_axes([0.85, 0.255, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 7, 8], [6, 7, 8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[g,y],
    node_size=5,dyad_lw=2)
ax.annotate("Techno-economic\noverlap", xy=(0.96, 0.24), 
            fontsize=12, ha='center')






# Line 5
ax_hypergraph = fig.add_axes([0.3, 0.1, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 2, 3], [1, 3], [2, 3],
    ],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[b],
    node_size=15,dyad_lw=3)
ax.annotate("(0.280)", xy=(0.37, 0.00), 
            fontsize=13, ha='center')
ax.annotate("Species", xy=(0.245, 0.15), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Climate", xy=(0.245, 0.01), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Effluent", xy=(0.377, 0.055), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Ocean\ncarbon-cycle\ndynamics", xy=(0.31, 0.065), 
            fontsize=13, ha='center')

ax_hypergraph = fig.add_axes([0.5, 0.1, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#4th
     [4, 5], [4, 6, 7],
     [4, 5], [4, 5, 6], [4, 5, 6, 7], [4, 5, 7], [4, 6], [4, 7], [5, 6],],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[g,r,y,p],
    node_size=15,dyad_lw=3)
ax.annotate("(0.164)", xy=(0.64, 0.00), 
            fontsize=13, ha='center')
ax.annotate("People", xy=(0.568, 0.15), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Research", xy=(0.568, 0.01), 
            fontsize=13, ha='right', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Governance", xy=(0.47, 0.055), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Machine", xy=(0.655, 0.055), 
            fontsize=13, ha='left', 
            color='black', bbox=dict(boxstyle="round", edgecolor="black", facecolor="none", pad=0.5))
ax.annotate("Aggregation and\ninnovation", xy=(0.67, 0.14), 
            fontsize=12, ha='center')


ax_hypergraph = fig.add_axes([0.7, 0.1, 0.15, 0.15])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [#3rd
     [1, 2], [1, 2, 3], [1, 3], [2, 3],
     #4th
     [4, 5], [4, 6, 7],
     [4, 5], [4, 5, 6], [4, 5, 6, 7], [4, 5, 7], [4, 6], [4, 7], [5, 6],
    #inter
    [1,4], [3,7]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,node_labels=node,
    pos=pos,
    edge_fc=[b,g,r,y,p],
    node_size=15,dyad_lw=3)

ax_hypergraph = fig.add_axes([0.85, 0.18, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 6, 7], [6, 7, 8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[r,y],
    node_size=5,dyad_lw=2)
ax.annotate("Socio-technic\noverlap", xy=(0.96, 0.145), 
            fontsize=12, ha='center')

ax_hypergraph = fig.add_axes([0.85, 0.103, 0.04, 0.04])  # Lowered bottom to give space
H = xgi.Hypergraph()
H.add_edges_from(
    [[5, 6, 8]],)

pos = xgi.circular_layout(H)
xgi.draw(H,hull=True,radius=0.15,
    pos=pos,
    edge_fc=[p],
    node_size=5,dyad_lw=2)
ax.annotate("Scientific\ndevelopment", xy=(0.96, 0.045), 
            fontsize=12, ha='center')



# Add annotations
ax.annotate("Resources\ninput:", xy=(0.001, 0.92), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Fishing,\nwind energy,\nwave energy\nharvesting and\ncollecting", xy=(0.001, 0.825),
            fontsize=13, ha='left')

ax.annotate("Appropriation\nand\ncirculation:", xy=(0.001, 0.7), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Port operations,\nshipbuilding,\nfreight shipping", xy=(0.001, 0.645),
            fontsize=13, ha='left')

ax.annotate("Transformation\nand\nconservation:", xy=(0.001, 0.51), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Marine protected\nareas,coastal\ninfrastructure,\naquaculture\ndesalination", xy=(0.001, 0.42),
            fontsize=13, ha='left')

ax.annotate("Consumption\nand\nexcretion:", xy=(0.001, 0.315), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Ship emissions,\nhydrocarbon\nextraction,\nland reclamation", xy=(0.001, 0.24),
            fontsize=13, ha='left')

ax.annotate("Wastes\noutput:", xy=(0.001, 0.14), weight='bold',
            fontsize=13, ha='left')
ax.annotate("Carbon sequestration,\nland-based effluent,\nwastewater treatment", xy=(0.001, 0.08),
            fontsize=13, ha='left')



ax.annotate("Oceanographic\nprocess", xy=(0.3, 0.97), weight='bold',
            fontsize=13, ha='center')
ax.annotate("Social\nprocess", xy=(0.58, 0.97), weight='bold',
            fontsize=13, ha='center')
ax.annotate("Socio-oceanographic\nprocess", xy=(0.83, 0.97), weight='bold',
            fontsize=13, ha='center')

# Save the plot as a PDF
pdf_file = "/Users/gurumakaza/Library/CloudStorage/OneDrive-MacauUniversityofScienceandTechnology/D盘/【实验】文章/【文章】MCIGI/2024年02月04日元模型/Figures/Fig6.pdf"
plt.savefig(pdf_file, format="pdf", bbox_inches="tight")  # Save as PDF with tight layout

# Show the plot
plt.tight_layout()

plt.show()














