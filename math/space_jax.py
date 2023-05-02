import numpy as np
import matplotlib.pyplot as py
from functools import partial
import jax.numpy as jnp
import jax
from jax import tree_util

class Grid:
    def __init__(self,ns,Ls,origin=0):
        ndim = len(ns)
        self.ns, self.Ls = jnp.asarray(ns).astype(int),jnp.asarray(Ls)
        dxs = jnp.asarray([Ls[i]/ns[i] for i in range(ndim)])
        self.dxs = dxs
        
        # origin of grid
        if origin == 0:
            origin = jnp.asarray([0.,0.,0.])
        
        # center of grid
        self.center = jnp.asarray([origin[i] + Ls[i]/2 for i in range(ndim)])

        # building the grid
        self.xs = jnp.asarray([jnp.linspace(origin[i],origin[i]+Ls[i],ns[i],endpoint=False) for i in range(ndim)])
        self.xxs = jnp.meshgrid(*self.xs,indexing='ij')

        # for 1D grids, it's often convenient to have short hand names
        if len(ns)==1:
            self.n, self.L, self.dx = self.ns[0], self.Ls[0], self.dxs[0]
            self.x = self.xs[0]

# register Grid class to a pytree node
tree_util.register_pytree_node(Grid, \
             lambda g: ((g.ns,g.Ls,g.dxs,g.xs,g.xxs,g.center), None), \
             lambda _, xs: Grid(xs[0], xs[1])) 

# differential operators in finite difference
partial(jax.jit,static_argnums=1)
def grad(phi,grid):
    return jnp.stack(jnp.gradient(phi,*grid.dxs),axis=-1)

# differential operators for pbc systems
partial(jax.jit,static_argnums=1)
def diff_fft(phi,grid,axis):
    phi_k = jnp.fft.fftn(phi)
    ks = [jnp.fft.fftfreq(n,dx)*2*jnp.pi for n,dx in zip(grid.ns,grid.dxs)]
    kks = jnp.meshgrid(*ks,indexing='ij')
    diff_k = 1j*kks[axis]

    return jnp.fft.ifftn(phi_k*diff_k).real

partial(jax.jit,static_argnums=1)
def grad_fft(phi,grid):
    ndim = len(grid.ns)
    return jnp.stack([diff_fft(phi,grid,i) for i in range(ndim)],axis=-1)

partial(jax.jit,static_argnums=1)
def div_fft(A,grid):
    n = A.shape[-1]
    return jnp.sum(jnp.asarray([diff_fft(A[...,i],grid,axis=i) for i in range(n)]),axis=0)

partial(jax.jit,static_argnums=1)
def inv_lapl_fft(phi,grid):
    ndim = len(grid.ns)
    phi_k = jnp.fft.fftn(phi)

    # diff kernel
    ks = jnp.asarray([jnp.fft.fftfreq(grid.ns[i],grid.dxs[i])*2*jnp.pi for i in range(ndim)])
    kks = jnp.meshgrid(*ks,indexing='ij')
    diff_ks = jnp.asarray([1j*kk for kk in kks])
    lapl_kernel = jnp.sum(jnp.asarray([diff_k**2 for diff_k in diff_ks]),axis=0)
    lapl_kernel = lapl_kernel.at[lapl_kernel==0].set(jnp.inf)

    # result_k
    R_k = phi_k/lapl_kernel
    
    # inversion and return
    return jnp.fft.ifftn(R_k).real


def int_fft(phi,grid,axis=0):
    phi_k =jnp.fft.fftn(phi)
    ks = [np.fft.fftfreq(n,dx)*2*np.pi for n,dx in zip(grid.ns,grid.dxs)]
    kks =jnp.meshgrid(*ks,indexing='ij')
    diff_k = 1j*kks[axis]
    diff_k[diff_k==0] =jnp.inf

    return jnp.fft.ifft(phi_k/diff_k).real


