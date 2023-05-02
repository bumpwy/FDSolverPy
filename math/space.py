import numpy as np
import matplotlib.pyplot as py

class Grid:
    def __init__(self,ns,Ls,origin=0):
        self.ns, self.Ls = np.asarray(ns), np.asarray(Ls)
        self.dxs = np.asarray([L/n for L,n in zip(Ls,ns)])
        
        # origin of grid
        if origin == 0:
            origin = np.asarray([0,0,0])
        
        # center of grid
        self.center = np.asarray([o + L/2 for o,L in zip(origin,Ls)])

        # building the grid
        self.xs = np.asarray([np.mgrid[0:L:dx]+origin \
                         for L,dx,origin in zip(self.Ls,self.dxs,origin)])
        self.xxs = np.meshgrid(*self.xs,indexing='ij')

        # for 1D grids, it's often convenient to have short hand names
        if len(ns)==1:
            self.n, self.L, self.dx = self.ns[0], self.Ls[0], self.dxs[0]
            self.x = self.xs[0]
        
# differential operators in finite difference
def grad(phi,grid):
    return np.stack(np.gradient(phi,*grid.dxs),axis=-1)

# differential operators for pbc systems
def diff_fft(phi,grid,axis=0):
    phi_k = np.fft.fftn(phi)
    ks = [np.fft.fftfreq(n,dx)*2*np.pi for n,dx in zip(grid.ns,grid.dxs)]
    kks = np.meshgrid(*ks,indexing='ij')
    diff_k = 1j*kks[axis]

    return np.fft.ifftn(phi_k*diff_k).real

def grad_fft(phi,grid):
    n = len(phi.shape)
    return np.stack([diff_fft(phi,grid,i) for i in range(n)],axis=-1)

def div_fft(A,grid):
    n = A.shape[-1]
    return np.sum([diff_fft(A[...,i],grid,axis=i) for i in range(n)],axis=0)

def inv_grad_fft(A,grid):
    n = A.shape[-1]
    A_ks = [np.fft.fftn(A[...,i]) for i in range(n)]

    # diff kernel
    ks = [np.fft.fftfreq(n,dx)*2*np.pi for n,dx in zip(grid.ns,grid.dxs)]
    kks = np.meshgrid(*ks,indexing='ij')
    diff_ks = [1j*kk for kk in kks]
    for diff_k in diff_ks:
        diff_k[diff_k==0] = np.inf

    # inversion
    phi_k = np.sum([A_k/diff_k for A_k,diff_k in zip(A_ks,diff_ks)],axis=0)

    return np.fft.ifftn(phi_k).real

def inv_div_fft(phi,grid):
    n = len(phi.shape)
    phi_k = np.fft.fftn(phi)

    # diff kernel
    ks = [np.fft.fftfreq(n,dx)*2*np.pi for n,dx in zip(grid.ns,grid.dxs)]
    kks = np.meshgrid(*ks,indexing='ij')
    diff_ks = [1j*kk for kk in kks]
    for diff_k in diff_ks:
        diff_k[diff_k==0] = np.inf

    # inversion
    A_ks = [phi_k/diff_k for diff_k in diff_ks]
    As = [np.fft.ifftn(A_k).real for A_k in A_ks]

    return np.stack(As,axis=-1)

def inv_lapl_fft(phi,grid):
    n = len(phi.shape)
    phi_k = np.fft.fftn(phi)

    # diff kernel
    ks = [np.fft.fftfreq(n,dx)*2*np.pi for n,dx in zip(grid.ns,grid.dxs)]
    kks = np.meshgrid(*ks,indexing='ij')
    diff_ks = [1j*kk for kk in kks]
    lapl_kernel = np.sum(np.asarray([diff_k**2 for diff_k in diff_ks]),axis=0)
    lapl_kernel[lapl_kernel==0] = (np.inf)

    # result_k
    R_k = phi_k/lapl_kernel
    
    # inversion and return
    return np.fft.ifftn(R_k).real
    #return np.fft.ifftn(phi_k/lapl_kernel).real


def int_fft(phi,grid,axis=0):
    phi_k = np.fft.fftn(phi)
    ks = [np.fft.fftfreq(n,dx)*2*np.pi for n,dx in zip(grid.ns,grid.dxs)]
    kks = np.meshgrid(*ks,indexing='ij')
    diff_k = 1j*kks[axis]
    diff_k[diff_k==0] = np.inf

    return np.fft.ifft(phi_k/diff_k).real


