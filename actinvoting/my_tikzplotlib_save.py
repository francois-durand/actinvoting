import re
from pathlib import Path

import ipynbname
import numpy as np


# Workarounds to make tikzplotlib work despite the deprecation of the package
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pgf
matplotlib.backends.backend_pgf.common_texification = matplotlib.backends.backend_pgf._tex_escape
np.float_ = np.float64
import matplotlib.legend
def get_legend_handles(legend):
    return legend.legend_handles
matplotlib.legend.Legend.legendHandles = property(get_legend_handles)
import webcolors

def integer_rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

webcolors.CSS3_HEX_TO_NAMES = {integer_rgb_to_hex(webcolors.name_to_rgb(name)): name
                               for name in webcolors.names("css3")}

import tikzplotlib
# End of the workarounds


def tikzplotlib_fix_ncols(obj):
    """Workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib

    Cf. https://stackoverflow.com/questions/75900239/attributeerror-occurs-with-tikzplotlib-when-legend-is-plotted
    """
    if hasattr(obj, "_ncols"):
        # noinspection PyProtectedMember
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def my_tikzplotlib_save(tikz_file_name, tikz_directory='sav', axis_width=r'\axisWidth', axis_height=r'\axisHeight',
                        invert_legend_order=False):
    """Save a figure in tikz.

    Parameters
    ----------
    tikz_file_name: str or Path
        Name of the tikz file.
    tikz_directory: str or Path
        Directory where to save the tikz file.
    axis_width: str
        Width of the axis in pgfplots.
    axis_height: str
        Height of the axis in pgfplots.
    invert_legend_order: bool
        If True, invert the order of the curves in the legend.
    """
    tikzplotlib_fix_ncols(plt.gcf())
    tikz_directory = Path(tikz_directory)
    if tikz_file_name is None:
        notebook_file_name = ipynbname.name()
        tikz_file_name = f'{notebook_file_name}.tex'
    tikz_directory.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(tikz_directory / tikz_file_name, axis_width=axis_width, axis_height=axis_height)
    with open(tikz_directory / tikz_file_name, 'r') as f:
        file_data = f.read()
    # Set 'fill opacity' of the legend to 1
    file_data = file_data.replace('fill opacity=0.8,', 'fill opacity=1,')
    # Set the font of the legend
    file_data = file_data.replace('legend style={', r'legend style={font=\legendFont, ')
    # Add yticks as they are in the matplotlib plot
    file_data = file_data.replace(
        'ytick style={',
        'ytick={' + ', '.join([str(y) for y in plt.yticks()[0]]) + '},\n'
        + 'ytick style={'
    )
    # Workaround in case of log scale
    if 'xmode=log' in file_data:
        file_data = re.sub(r'default{10\^{.*?}}', '', file_data)
        file_data = re.sub(r'(?s)xtick={.*?},.*?xticklabels={.*?}', 'xmode=log', file_data)
    if 'ymode=log' in file_data:
        file_data = re.sub(r'default{10\^{.*?}}', '', file_data)
        file_data = re.sub(r'(?s)ytick={.*?},.*?yticklabels={.*?}', 'ymode=log', file_data)
    # Prevent from scaling down the plt.text
    file_data = file_data.replace('scale=0.5,', 'scale=1.0,')
    # Invert legend order if asked
    if invert_legend_order:
        file_data = file_data.replace(r'\begin{axis}[', r'\begin{axis}[reverse legend,')
    with open(tikz_directory / tikz_file_name, 'w') as f:
        f.write(file_data)
