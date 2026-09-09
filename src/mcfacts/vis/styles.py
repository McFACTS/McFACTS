import matplotlib.colors as mcolors

# Colors
color_gen1 = "darkgoldenrod"
color_gen2 = "rebeccapurple"
color_genX = "red"
color_line1 = "teal"

# Marker Styles
marker_gen1 = 'o'
marker_gen2 = 'v'
marker_genX = '^'

# Marker Size
markersize_gen1 = 10
markersize_gen2 = 10
markersize_genX = 10

# Marker transperancy
markeralpha_gen1 = 0.6
markeralpha_gen2 = 0.6
markeralpha_genX = 0.6

gen_styles = {
    "g1": dict(marker=marker_gen1, color=color_gen1,
               rgb=mcolors.to_rgb(color_gen1),
               size=0.4 * markersize_gen1, alpha=markeralpha_gen1,
               label="1g-1g"),
    "g2": dict(marker=marker_gen2, color=color_gen2,
               rgb=mcolors.to_rgb(color_gen2),
               size=0.4 * markersize_gen2, alpha=markeralpha_gen2,
               label="2g-1g or 2g-2g"),
    "gX": dict(marker=marker_genX, color=color_genX,
               rgb=mcolors.to_rgb(color_genX),
               size=0.4 * markersize_genX, alpha=markeralpha_genX,
               label=r"$\geq$3g-Ng"),
}
