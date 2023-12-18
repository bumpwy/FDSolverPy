import numpy as np
from scipy.constants import physical_constants
import subprocess
import itertools as it
import more_itertools as mit

kB = physical_constants['Boltzmann constant in eV/K'][0]


# read diffusivity parameter file
def read_block(f,before_block,in_block):
    block = []
    for line in it.takewhile(in_block,it.dropwhile(before_block,f)):
        block += [line.strip()]
    return block
    
def read_diff(fname):
    def before_block(line): return not line.startswith('&')
    def in_block(line): return line!=('/\n')
    
    with (open(fname,'r')) as f:
        dat = {}
        pf = mit.peekable(f)
        while True:
            block = read_block(pf,before_block,in_block)
            dat[block[0][1:]] = np.array([list(map(float,line.split())) for line in block[1:]])
            if pf.peek('eof')=='eof': break
    return dat
def calc_diffusivity(fname,T):
    kB = 8.617333262E-5
    dat = read_diff(fname)
    Db = dat['bulk_D0']*np.exp(-dat['bulk_Ea']/kB/T)     
    Dgb = dat['gb_D0']*np.exp(-dat['gb_Ea']/kB/T)     
    return Db, Dgb
def calc_diffusivity_hti(T):
    # temperature
    kT = kB*T
    
    # equilibrium occupancies
    e_o, e_t = 0, 0.1
    Z = np.exp(-e_o/kT) + np.exp(-e_t/kT)
    r_o, r_t = np.exp(-e_o/kT)/Z, np.exp(-e_t/kT)/Z

    # transition rates
    v = 4E13
    # bulk
    e_oo, e_tt, e_ot, e_to =  0.69, 10, 0.49, 0.39  # e_tt is just a big number
    l_oo, l_tt = v*np.exp(-e_oo/kT), v*np.exp(-e_tt/kT)
    l_ot, l_to = v*np.exp(-e_ot/kT), v*np.exp(-e_to/kT)
    # surface
    e_f, e_h = 0, 0.03
    Z_s = np.exp(-e_f/kT) + np.exp(-e_h/kT)
    r_f = np.exp(-e_f/kT)/Z_s
    e_fh = 0.24
    l_fh = v*np.exp(-e_fh/kT)

    # diffusivity tensor
    a,c = 2.933/1e8, 4.638/1e8  # in cm units
    Db = (4*r_t*l_to/2) * a**2
    Dc = (2*r_o*l_oo/4 + 4*r_t*(3*l_to*l_tt)/(24*l_to+16*l_tt)) * (c/2)**2
    Dgb = 3 * (a/np.sqrt(3))**2 * r_f * l_fh

    return np.array([[Db,0,0],\
                     [0,Db,0],\
                     [0,0,Dc]]),\
           np.array([[Dgb,0,0],\
                     [0,Dgb,0],\
                     [0,0,Dgb]])

def calc_diffusivity_SrGDC(T):
    # temperature
    kT = kB*T
    
    D0_b, D0_gb, D0_surf = [8.9]*3
    E_b, E_gb, E_surf = 6.23, 4.83, 2.9

    D_b, D_gb, D_surf = D0_b * np.exp(-E_b/kT),\
                        D_gb * np.exp(-E_gb/kT),\
                        D_surf * np.exp(-E_surf/kT)
    return np.diag([D_b]*3),\
           np.diag([D_gb]*3),\
           np.diag([D_surf]*3)


