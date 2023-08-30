import numpy as np
import subprocess
import itertools as it
import more_itertools as mit

def read_file(fname):
    dat = np.fromfile(fname,sep=' ')
    phi = dat[3::4]
    n = round(len(phi)**(1/3))
    phi = phi.reshape((n,n,n))
    return phi

def set_diffusivity(phi,Db,Dgb):
    return phi*Db + (1-phi)*Dgb

def read_microstruct(header):
    
    # info file
    lines = open(f'{header}.info').read().strip().split('\n')
    ns = [int(x) for x in lines[0].split()[1::2]]
    
    # microstructure file
    d = np.array(open(f'{header}.in','rt').read().strip().split('\n')).astype(float)
    nq = int(d.size/ns[0]/ns[1]/ns[2])
    zetas = d.reshape((nq,*ns),order='F')

    return zetas

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
    kB = 8.617333262E-5
    kT = kB*T
    
    # equilibrium occupancies
    e_o, e_t = 0, 0.1
    Z = np.exp(-e_o/kT) + np.exp(-e_t/kT)
    r_o, r_t = np.exp(-e_o/kT)/Z, np.exp(-e_t/kT)/Z

    # transition rates
    v = 4E13
    e_oo, e_tt, e_ot, e_to =  0.69, 10, 0.49, 0.39  # e_tt is just a big number
    l_oo, l_tt = v*np.exp(-e_oo/kT), v*np.exp(-e_tt/kT)
    l_ot, l_to = v*np.exp(-e_ot/kT), v*np.exp(-e_to/kT)

    # diffusivity tensor
    #a,c = 2.933, 4.638
    a,c = 2.933, 4.638/2
    Db = (4*r_t*l_to/2) * a**2
    Dc = (2*r_o*l_oo/4 + 4*r_t*(3*l_to*l_tt)/(24*l_to+16*l_tt)) * c**2

    return np.array([[Db,0,0],\
                     [0,Db,0],\
                     [0,0,Dc]])





