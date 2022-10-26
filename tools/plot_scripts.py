import numpy as np
import subprocess
from mpl_toolkits import mplot3d
import matplotlib.pyplot as py
from matplotlib import cm

# plotting cuboid
def draw_cuboid(XXs,scalar,fig=None,ax=None,plotter='mayavi',cmap=None,**kwargs):
    Ns = XXs[0].shape
    if plotter=='matplotlib':
        if fig is None: fig = py.figure()
        if ax is None: ax = fig.add_subplot(projection='3d')
        cmap_func = cm.get_cmap(cmap)
        cmap_func.set_bad(color='black')
        smin,smax = scalar.min(), scalar.max()
    plots = []
    for i in range(3):
        N = Ns[i]
        L = XXs[i].max()
        for s in range(2):
            inds = tuple([s*(N-1) if j==i else np.s_[:] for j in range(3)])
            grids = [np.zeros_like(scalar[inds])+s*L if j==i else XXs[j][inds] for j in range(3)]
            if plotter=='mayavi':
                plots += [mlab.mesh(*grids,scalars=scalar[inds],**kwargs)]
            elif plotter=='matplotlib':
                my_col = cmap_func((scalar[inds]-smin)/(smax-smin))
                plots += [ax.plot_surface(*grids,facecolors=my_col,edgecolor='none',**kwargs)]
    if plotter=='mayavi': return plots
    elif plotter=='matplotlib': return ax, plots
    
def plot_Dij(Ds,ax,args):
    i_flat = range(9)
    for D in Ds:
        ax.plot(i_flat,D.ravel(),'o',**args)
    ax.set_xticks(i_flat)
    ax.set_xticklabels([r'$D_{%i%i}$'%(i+1,j+1) for i,j in zip(*np.unravel_index(i_flat,(3,3)))])
