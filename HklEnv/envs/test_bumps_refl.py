import bumps.names as bumps 
import bumps.fitters as fitters

import numpy as np
import sys
import os

from .find_min import findmin

def config_matplotlib(backend=None):
    """
    Setup matplotlib to use a particular backend.

    The backend should be 'WXAgg' for interactive use, or 'Agg' for batch.
    This distinction allows us to run in environments such as cluster computers
    which do not have wx installed on the compute nodes.

    This function must be called before any imports to pylab.  To allow
    this, modules should not import pylab at the module level, but instead
    import it for each function/method that uses it.  Exceptions can be made
    for modules which are completely dedicated to plotting, but these modules
    should never be imported at the module level.
    """

    # When running from a frozen environment created by py2exe, we will not
    # have a range of backends available, and must set the default to WXAgg.
    # With a full matplotlib distribution we can use whatever the user prefers.
    if hasattr(sys, 'frozen'):
        if 'MPLCONFIGDIR' not in os.environ:
            raise RuntimeError(
                r"MPLCONFIGDIR should be set to e.g., %LOCALAPPDATA%\YourApp\mplconfig")
        if backend is None:
            backend = 'WXAgg'

    import matplotlib
    from matplotlib import pyplot

    # Specify the backend to use for plotting and import backend dependent
    # classes. Note that this must be done before importing pyplot to have an
    # effect.  If no backend is given, let pyplot use the default.
    if backend is not None:
        matplotlib.use(backend)

    # Disable interactive mode so that plots are only updated on show() or
    # draw(). Note that the interactive function must be called before
    # selecting a backend or importing pyplot, otherwise it will have no
    # effect.

    #matplotlib.interactive(True)

    #configure the plot style
    line_width = 1
    pad = 2
    font_family = 'Arial' if os.name == 'nt' else 'sans-serif'
    font_size = 12
    plot_style = {
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': line_width,
        'axes.linewidth': line_width,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.major.width': line_width,
        'ytick.major.width': line_width,
        'xtick.minor.width': line_width,
        'ytick.minor.width': line_width,
        'xtick.major.pad': pad,
        'ytick.major.pad': pad,
        'xtick.top': True,
        'ytick.right': True,
        'font.size': font_size,
        'font.family': font_family,
        'svg.fonttype': 'none',
        'savefig.dpi': 100,
    }
    matplotlib.rcParams.update(plot_style)

def better_bumps(model):

    #config_matplotlib("TkAgg")
    config_matplotlib("Agg")
    zin=[]
    zout=[]
    chis=[]
    dzs=[]
    nllfs=[]

    for zs in np.arange(.05,.45,.005):
        #print("zs", zs)
        model.atomListModel.atomModels[0].z.value = zs
        model.update()
        schi=model.nllf()
        nllfs.append(schi)
        zin.append(zs)
    xpeaks = findmin(zin, nllfs, 10)
    model.atomListModel.atomModels[0].z.value = zin[xpeaks[0]]
    model.update()
    problem = bumps.FitProblem(model)

    result = fitters.fit(problem, method='lm')
    #from matplotlib import pyplot as plt
    #problem.plot()
    #plt.show()

    for p, v in zip(problem._parameters, result.dx):
        p.dx = v
    
    return result.x, result.dx, problem.chisq(), problem._parameters
    






