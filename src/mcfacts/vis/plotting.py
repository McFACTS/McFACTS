
# Dictionary of widths in points for different LaTeX  journal templates
figure_sizes = {"aa_col": 256.0748,
                "aa_page": 523.5307,
                "mnras_col": 244.0,
                "mnras_page": 508.0,
                "apj_col": 242.2665,
                "apj_page": 513.1174}


def set_size(width, fraction=1, subplots=(1, 1), square=False):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width : float or string
        Document width in points, or string of predefined document type
    fraction : float, optional
        Fraction of the width which you wish the figure to occupy
    subplots : array-like, optional
        The number of rows and columns of subplots
    square : bool
        Whether or not figure should be square
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if isinstance(width, str):
        width_pt = figure_sizes[width]
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if (square is True):
        fig_height_in = fig_width_in
    else:
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
