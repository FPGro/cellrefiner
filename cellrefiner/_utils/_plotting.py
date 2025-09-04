from typing import Optional,  Tuple
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from ..tools._cell_shape_modeling import SEM

def get_axes(ax: Optional[Axes] = None) -> Tuple[Figure, Axes]:
    """create or get axes"""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def get_cid_list(sem: SEM, cid_list: Optional[np.ndarray], scaling=True):
    if scaling:
        xe = sem.xe*sem.scale+sem.deltax
    else:
        xe = sem.xe

    if cid_list is None:
        cid_list = np.arange(sem.nc)
    else:
        xe_vis = []
        for cid in cid_list:
            xe_vis.append(xe[sem.ceidn[cid]:sem.ceidn[cid+1]])
        xe = np.vstack(xe_vis)
    return cid_list, xe


def add_colorbar(fig, ax, cmap, norm):
    """add colorbar"""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    divider = make_axes_locatable(ax)
    # to do fix colorbar width
    cax = divider.append_axes("right", size="5%", pad=0.4)
    cb = plt.colorbar(sm, cax=cax)
    return cb


def set_axes(ax, show_axis):
    """aspect equal, axis off, invert yaxis"""
    ax.set_aspect('equal', adjustable='box')
    ax.autoscale(tight=True)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if not show_axis:
        ax.set_axis_off()


def get_arr(sem, vis_key, arr, summary):
    if (sem.adata is not None) & (vis_key is not None):
        if summary == 'gene' and vis_key in sem.adata.var_names:
            # retrieve gene expression
            arr = sem.adata[:, vis_key].X.toarray()[:, 0]
        elif summary == 'sender' and 'sender_signal' in sem.adata.obsm and vis_key in sem.adata.obsm['sender_signal']:
            # retrieve sender signal
            arr = sem.adata.obsm['sender_signal'][vis_key].to_numpy()
        elif summary == 'receiver' and 'receiver_signal' in sem.adata.obsm and vis_key in sem.adata.obsm['receiver_signal']:
            # retrieve receiver signal
            arr = sem.adata.obsm['receiver_signal'][vis_key].to_numpy()
        elif vis_key in sem.adata.obs:
            arr = sem.adata.obs[vis_key]  # retrieve adata.obs
    return arr


def get_cat_arr_color(sem, arr, cid_list, vis_key, cmap_name):
    cat_code = arr.cat.codes[cid_list]
    cat_list = arr.cat.categories
    if (vis_key+'_colors') in sem.adata.uns:
        # use cluster color in the adata
        color_list = sem.adata.uns[vis_key+'_colors']
        if type(color_list[0]) is str:
            color_list = np.array([to_rgb(x) for x in color_list])
    else:
        cmap = colormaps[cmap_name]
        color_list = cmap(np.linspace(0, 1, len(cat_list)))[:, :3]
    return cat_code, cat_list, color_list