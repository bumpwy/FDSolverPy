# WKC.py 
# WKC is the class that evolves the WKC 
# phase-field equations. 
#############################################################
import numpy as np
from scipy.optimize import line_search
import opt_einsum as oe
import os, subprocess, pickle, gc
import sys
sys.path.append('../')
from FDSolverPy.base.ParallelSolver import *
from FDSolverPy.math.space import *
from datetime import datetime
import itertools as it


class diff_solver(parallel_solver):
    def __init__(self,
                 # inputs for grid
                 Xs,ghost=2,
                 # pbc
                 pbc=(0,0,0),
                 # diffusivity
                 D='D.npz',
                 # variable initialization
                 C='C.npz',Data_Type=['double'],
                 # extra (mostly for backward compatability)
                 **kwargs):
        # call parent constructor
        parallel_solver.__init__(self,Xs,ghost,pbc=pbc)
        
        # load variables
        if type(D) is str: 
            D_array = np.load(D)['D']
        else:
            D_array = D
        if type(C) is str: 
            C_array = np.load(C)['C']
        else:
            C_array = C
        
        # setup variables
        self.d = np.zeros(tuple(self.nes+[self.ndim,self.ndim]),dtype=D_array.dtype)
        self.c = np.zeros(tuple(self.nes),dtype=C_array.dtype)
        c_mpi_type = MPI._typedict[self.c.dtype.char]
        self.set_variables(varnames=['c'],dat=[self.c],\
                           dat_bc=[None],dat_type=[c_mpi_type])
        
        # distribute large grid to decomposed grid (for each cpu)
        self.distribute(self.d,D_array)
        self.distribute(self.c,C_array)

        # for speed
        ops = self.d,tuple(self.nes+[self.ndim])
        self.J_oe_expr = oe.contract_expression('abcij,abcj->abci', *ops, constants=[0])
        self.e_oe_expr = oe.contract_expression('abci,abci->abc',\
                                                tuple(self.nes+[self.ndim]),\
                                                tuple(self.nes+[self.ndim]))

        # parameters
        self.C = C
        # store parameters
        self.dict = {'Xs':Xs,'ghost':ghost,'pbc':pbc,
                     'D':D, 'C':C, 'Data_Type':Data_Type}
    def run(self,outdir='data',restart=False,
            Nstep=100,step=1,etol=1e-4,ftol=1e-4,ls_args={}):
        
        ########## Initialize Run ##########
        # setup file pointers, loading last frame, etc.
        self.initialize_run(outdir,restart=restart)
        # setup counters
        Nstep,step = int(Nstep), int(step)
        if restart:
            # load counter variable at root process
            if self.rank == 0:
                [counter] = self.counter_mm.tolist()
            else:
                counter = None
            # sync counter across all processes
            counter = self.comm.bcast(counter,root=0)
            Nstep += counter
        else:
            counter = 0
            self.dump(outdir,counter) # store initial frame if it's a fresh start
        # store the class object as dict
        if self.rank == 0:
            pickle.dump(self.dict,
                        open('diff_solver.pckl','wb'),
                        protocol=-1)
        
        ########## Optimization Setup ##########
        #self.str_to_alg(alg)
        hh,g0,g1 = [np.zeros_like(self.c) for i in range(3)]
        Fe0,Err = self.dF(self.c,g0)
        hh[:] = g0
        Fe_old = 0
        DF = np.Inf
        ############### The Big Loop ###############
        t1 = datetime.now()
        self.parprint('Big Loop Begins...')
        self.parprint("%s%s%s%s%s"\
                       %('it(#)'.ljust(10),'F(eV)'.ljust(15),'dF(eV)'.ljust(25),\
                         'Force_max(eV/A)'.ljust(20),'Time(h:m:s)'.ljust(15)))
        self.parprint("%s%s%s%s%s"\
                        %(('%i'%counter).ljust(10),\
                          ('%.4e'%Fe0).ljust(15),\
                          (f'').ljust(25),\
                          ('%.4e'%Err).ljust(20),\
                          str(datetime.now()-t1).ljust(15)))
        
        while (Err > ftol or DF > etol) and counter < Nstep:
            # - MPI synchronization
            self.comm.Barrier()
            gc.collect()
            
            # - integration
            Fe_old = Fe0
            Fe0,Err = self.cg_integrate(self.c,hh,g0,g1,ls_args)

            # - record results
            if Fe_old==0: DF=0
            else:DF = np.absolute((Fe0-Fe_old)/Fe_old)
            counter += 1
            
            # - output data
            if counter%step==0:
                self.dump(outdir,counter)
                self.parprint("%s%s%s%s%s"\
                                %(('%i'%counter).ljust(10),\
                                  ('%.4e'%Fe0).ljust(15),\
                                  (f'{(Fe0-Fe_old):.4e}/{(100*DF):.4f} %').ljust(25),\
                                  ('%.4e'%Err).ljust(20),\
                                  str(datetime.now()-t1).ljust(15)))
        # final output
        self.dump(outdir,counter)
        self.parprint("%s%s%s%s%s"\
                        %(('%i'%counter).ljust(10),\
                          ('%.4e'%Fe0).ljust(15),\
                          (f'{(Fe0-Fe_old):.4e}/{(100*DF):.4f} %').ljust(25),\
                          ('%.4e'%Err).ljust(20),\
                          str(datetime.now()-t1).ljust(15)))
        self.parprint('Big Loop time lapse: %s'%(str(datetime.now()-t1)))
        ############### The Big Loop ###############

    # relaxation algorithms
    def cg_integrate(self,d,hh,g0,g1,ls_args={}):
       
        ###### The Line Search #####
        #t = self.golden_line_search(hh,d,**ls_args)
        ts, Fs = self.bracket(0,1e-5,d,hh)
        t = self.brent_line_search(ts, Fs, d, hh, **ls_args)
        d -= t*hh
        
        ###### Construct New Conjugate Direction ######
        Fe0,Err = self.dF(d,g1)
        alpha = self.cg_coefficient(g0,g1)
        hh[:] = alpha*hh + g1
        g0[:] = g1
        
        return Fe0,Err

    def cg_coefficient(self,g0,g1):
        nom = ((g1-g0)*g1)[self.ind]
        denom = (g0**2)[self.ind]
        alpha1=self.par_sum(nom)
        alpha2=self.par_sum(denom)
        if alpha2==0: return 0
        return max(alpha1/alpha2, 0)

    def bracket(self,ta,tb,x,d):
        gold,glim = (1+np.sqrt(5))/2, 100
        Fa,Fb = self.F(x-ta*d), self.F(x-tb*d)
        if Fb > Fa:
            ta, tb = tb, ta
            Fa, Fb = Fb, Fa
        tc = tb + gold*(tb-ta)
        Fc = self.F(x-tc*d)

        # iteratively determine tc
        while Fb > Fc:
            r, q = (tb-ta)*(Fb-Fc), (tb-tc)*(Fb-Fa)
            tu = tb-((tb-tc)*q - (tb-ta)*r)\
                     /(2*np.sign(q-r) * np.absolute(max(np.absolute(q-r),1e-20)))
            tulim = tb + glim*(tc-tb)
            if ((tb-tu)*(tu-tc)) > 0:
                Fu = self.F(x-tu*d)
                if Fu < Fc:
                    ta, tb = tb, tu
                    Fa, Fb = Fb, Fa
                    return (ta,tb,tc), (Fa,Fb,Fc)
                elif Fu > Fb:
                    tc, Fc = tu, Fu
                    return (ta,tb,tc), (Fa,Fb,Fc)
                tu = tc + gold*(tc-tb)
                Fu = self.F(x-tu*d)
            elif ((tc-tu)*(tu-tulim)) > 0:
                Fu = self.F(x-tu*d)
                if (Fu < Fc):
                    tb, tc, tu = tc, tu, tu+gold*(tu-tc)
                    Fb, Fc, Fu = Fc, Fu, self.F(x-tu*d)
            elif ((tu-tulim)*(tulim-tc))>=0:
                tu = tulim
                Fu = self.F(x-tu*d)
            else:
                tu = tc + gold*(tc-tb)
                Fu = self.F(x-tu*d)
            ta, tb, tc = tb, tc, tu
            Fa, Fb, Fc = Fb, Fc, Fu
        return (ta, tb, tc), (Fa, Fb, Fc)
    
    def brent_line_search(self,ts,Fs,x,d,tol=1e-4,maxiter=100):
        
        # consts
        gold, eps = 1/(np.sqrt(5)+1), 1e-40
        
        # initialize points
        a, b, c = ts
        Fa, Fb, Fc = Fs
        o, w, v = b, b, b
        Fw = Fv = Fo = self.F(x-o*d)
        e, g = 0, 0

        for i in range(maxiter):
            m = (a + b)*0.5
            tol1 = tol*np.absolute(o)+eps
            tol2 = 2*tol1
            if np.absolute(o-m) <= (tol2-0.5*(b-a)):
                Fmin = Fo
                return o
            if (np.absolute(e) > tol1):
                r, q = (o-w)*(Fo-Fv),(o-v)*(Fo-Fw)
                p = (o-v)*q - (o-w)*r
                q = 2*(q-r)
                if q>0:
                    p = -p
                q = np.absolute(q)
                etemp = e
                e = g
                if (np.absolute(p) >= np.absolute(0.5*q*etemp)) or \
                        (p <= q*(a-o)) or (p >= q*(b-o)):
                    e = (a-o if (o>=m) else b-o)
                    g = gold*e
                else:
                    g = p/q
                    u = o+g
                    if (u-a < tol2) or ((b-u)<tol2):
                        g = np.sign(m-o)*np.absolute(tol1)
            else:
                e = (a-o if o>=m else b-o)
                g = gold*e
            u = (o+g if (np.absolute(g) >= tol1) else o+np.sign(g)*np.absolute(tol1))
            Fu = self.F(x-u*d)
            if Fu <= Fo:
                if u>=o: a=o
                else: b=o
                v,w,o = w,o,u
                Fv,Fw,Fo = Fw,Fo,Fu
            else:
                if (u<o): a=u
                else: b=u
                if (Fu<=Fw) or (w==o):
                    v = w
                    w = u
                    Fv = Fw
                    Fw = Fu
                elif (Fu <= Fv) or v==o or v==w:
                    v, Fv = u, Fu
        return o




    def golden_line_search(self,dF_d,d,mu=1e-6,tol_l=1e-2):
        ###### The Golden Line Search #####
        # point 0 ...................
        alpha0 = 0
        Fe0 = self.F(d-alpha0*dF_d)
        # point 1 ...................
        alpha1 = mu
        Fe1 = self.F(d-alpha1*dF_d)
        #self.parprint('point 1...')
        while Fe1>Fe0:
            #self.parprint(Fe1)
            alpha1 /= 2
            Fe1 = self.F(d-alpha1*dF_d)
        # point 2 ...................
        alpha2 = alpha1*10
        Fe2 = self.F(d-alpha2*dF_d)
        #self.parprint('point 2...')
        while Fe2<Fe1:
            #self.parprint(Fe2)
            alpha2*=10
            Fe2 = self.F(d-alpha2*dF_d)
        # now we do the line search......
        h = alpha2-alpha0
        invphi = (np.sqrt(5)-1)/2
        invphi2 = (3-np.sqrt(5))/2
        n = int(np.ceil(np.log(tol_l/h)/np.log(invphi)))
        c = alpha0 + invphi2*h
        dd = alpha0 + invphi*h
        Fec = self.F(d-c*dF_d)
        Fed = self.F(d-dd*dF_d)
        #self.parprint('line search...')
        for k in range(n-1):
            #self.parprint(k,Fed)
            if Fec<Fed:
                alpha2 = dd
                dd = c
                Fed = Fec
                h *= invphi
                c = alpha0 + invphi2*h
                Fec = self.F(d-c*dF_d)
            else:
                alpha0 = c
                c = dd
                Fec = Fed
                h*=invphi
                dd = alpha0 + invphi*h
                Fed = self.F(d-dd*dF_d)
        if Fec < Fed:
            if Fe1<Fec:
                return alpha1
            else:
                return 0.5*(alpha0+dd)
        else:
            if Fe1<Fed:
                return alpha1
            else:
                return 0.5*(c+alpha2)
    
    def wolfe_line_search(self,dF_d,d,mask=None,t=1.,c1=1e-5,c2=0.01,a=0.,b=100.):
        dF0,dF1 = np.zeros_like(dF_d),np.zeros_like(dF_d)
        Fe0,Err0 = self.dF(d,dF0)
        fp0 = (self.comm.allreduce(sendobj=np.sum(dF_d*dF0),op=MPI.SUM))
        self.parprint(Fe0)
        while True:
            Fe1,Err1 = self.dF(d-t*dF_d,dF1)
            self.parprint(Fe1)
            fp1 = (self.comm.allreduce(sendobj=np.sum(dF_d*dF1),op=MPI.SUM))
            if Fe1 > (Fe0 - c1*t*fp0):
                b=t
                t=(a+b)/2
            elif fp1 > c2*fp0:
                a=t
                t=(a+b)/2
            else:
                break
        return t
    def backtracking_line_search(self,dF_d,Fe0,d,mask=None,t=1e-4,beta=0.5,c=1e-5):
        fp = (self.comm.allreduce(sendobj=np.sum(dF_d**2),op=MPI.SUM))
        while self.F(self.d-t*dF_d) > (Fe0 - c*t*fp): 
            t*=beta
        return t

    ##### energy & force #####
    def FirstDiff(self,A,axis):
        # first take gradient
        diffA = np.gradient(A,self.dxs[axis],axis=axis,edge_order=1)
        A_view = np.swapaxes(A,axis,0)
        diffA_view = np.swapaxes(diffA,axis,0)
        
        # handle edge terms
        diffA_view[0,...]  /= np.sqrt(2)
        diffA_view[-1,...] /= np.sqrt(2)

        return diffA

    def SecondDiff(self,A,axis):
        # first take gradient
        diffA = np.gradient(A,self.dxs[axis],axis=axis,edge_order=1)
        A_view = np.swapaxes(A,axis,0)
        diffA_view = np.swapaxes(diffA,axis,0)
        # handle edge terms
        diffA_view[0,...] =  ( 1.0*A_view[0,...]  + 0.5*A_view[1,...])/self.dxs[axis]
        diffA_view[1,...] =  (-1.0*A_view[0,...]  + 0.5*A_view[2,...])/self.dxs[axis]
        diffA_view[-2,...] = (-0.5*A_view[-3,...] + 1.0*A_view[-1,...])/self.dxs[axis]
        diffA_view[-1,...] = (-0.5*A_view[-2,...] - 1.0*A_view[-1,...])/self.dxs[axis]

        return diffA
 
    def F(self,c):
        # calculate energy
        self.update_boundary(c)
        #gradCs = np.stack(np.gradient(c,*self.dxs),axis=-1)
        gradCs = np.stack([self.FirstDiff(c,i) for i in range(self.ndim)],axis=-1)

        ##### NumPy #####
        J = self.J_oe_expr(gradCs)
        e_density = 0.5*self.e_oe_expr(J,gradCs)
         
        return self.par_sum((e_density[self.ind]))

    def dF(self,c,dF_dc,mask=None):
        # calculate energy
        self.update_boundary(c)
        gradCs = np.stack(np.gradient(c,*self.dxs),axis=-1)
        #gradCs = np.stack([self.FirstDiff(c,i) for i in range(self.ndim)],axis=-1)
       
        ##### NumPy approach #####
        J = self.J_oe_expr(gradCs)
        #e_density = 0.5*self.e_oe_expr(J,gradCs)
       
        # calculate force
        #dF_dc[:] = sum([-self.SecondDiff(J[...,i],i) for i in range(self.ndim)])
        dF_dc[:] = sum([-np.gradient(J[...,i],dx,axis=i) for i,dx in enumerate(self.dxs)])

        # here we apply a fix boundary condition
        self.fix_boundary(dF_dc)
        self.update_boundary(dF_dc)
        
        # maximal force
        e = np.sqrt(dF_dc**2).max()
         
        #return self.par_sum((e_density[self.ind])),\
        #       self.comm.allreduce(sendobj=e,op=MPI.MAX)
        return self.F(c),\
               self.comm.allreduce(sendobj=e,op=MPI.MAX)
    def fix_boundary(self,dF_dc):
        for i in range(self.ndim):
            if self.pbc[i]: continue
            df_dc = np.swapaxes(dF_dc,i,0)
            df_dc[0,...],df_dc[-1,...] = 0, 0



##### helper functions #####
def read_diffsolver_args(path='.'):
    # read in dictionary object
    dct = pickle.load(open(os.path.join(path,'diff_solver.pckl'),'rb'))
    return dct

def read_diffsolver_data(path='.',prefix='data',frame=-1):
    # read in dictionary object
    dct = read_diffsolver_args(path)
    Ns = [len(X) for X in dct['Xs']]

    data_path = os.path.join(path,prefix)
    
    # optimized concentration field
    if frame <0:
        counter_mm = np.memmap(os.path.join(data_path,'counter.dat'),\
                               dtype='int32',mode='r+',shape=(1,))
        [count] = counter_mm.tolist()
        count += (frame + 1)
    else:
        count = frame
    # make sure frame exists
    while not os.path.exists(os.path.join(data_path,f'c.{count}')):
        count -= 1
    print(f'Reading the {count}th frame...')
    C = np.fromfile(os.path.join(data_path,f'c.{count}'),dtype=np.double).reshape(Ns,order='F')

    return dct, C
    
def read_diffsolver(path='.',prefix='data',frame=-1):
    # read in data
    dct, c = read_diffsolver_data(path,prefix,frame)

    # diffsolver object
    cwd = os.getcwd()
    os.chdir(path)
    calc = diff_solver(**dct)
    os.chdir(cwd)

    # calculate current
    q = -np.stack(np.gradient(c,*calc.dxs),axis=-1)
    j = np.einsum('abcij,abcj->abci',calc.d,q)

    return calc, c, q[calc.ind], j[calc.ind]

# write inputs
def write_inputs(path,input_dct,C,D):
    subprocess.call(f'mkdir -p {path}',shell=True)
    pickle.dump(input_dct,\
                open(os.path.join(path,'diff_solver.pckl'),'wb'),\
                protocol=-1)
    np.savez_compressed(os.path.join(path,'C.npz'),C=C)
    np.savez_compressed(os.path.join(path,'D.npz'),D=D)

# create the initial concentration field according to Q-vector
# if Q is not provided, i.e. Q=None, a randomized unit vector will be used

def micro_delC(d,grid,Q):
    grad_inv_d = grad_fft(1/d,grid)
    f = np.einsum('abc,abci->abci',d,grad_inv_d)
    fQ = np.einsum('abci,i->abc',f,Q)

    # first order term
    del_c = -inv_lapl_fft(fQ,grid)

    # invert Poisson
    return del_c
def macro_C0(grid,Q=None):
    # create Q-vector if needed
    if Q is None:
        Q = np.random.rand(3)-0.5
        Q /= np.linalg.norm(Q)

    # macroscopic C
    C0 = np.einsum('i,iabc->abc',-Q,np.array(grid.xxs))

    return C0

def create_C(grid,d,Q=None):

    return macro_C0(grid,Q) + micro_delC(d,grid,Q)*0.4


# calculates effective diffusivity from given calculation results
# If three are given then spits out one D_ij, but if more are given
# it spits out results of all combinations of 3
def calculate_D(paths,prefixes='data'):
    if isinstance(prefixes,str):
        prefixes = [prefixes]*len(paths)

    # First, load all data and take mean
    Qs,Js = [], []
    cwd = os.getcwd()
    for path,prefix in zip(paths,prefixes):
        os.chdir(path)
        calc,c,q,j = read_diffsolver('.',prefix)
        J,Q = j.mean(axis=(0,1,2)), q.mean(axis=(0,1,2))
        Js += [J]
        Qs += [Q]
        os.chdir(cwd)
    
    # Second, pick 3 out of all results and calculate effective diffusivity
    n = len(paths)
    if n==3:
        Jn = np.stack(Js,axis=-1)
        Qn = np.stack(Qs,axis=-1)
        D = np.einsum('in,nj->ij',Jn,np.linalg.inv(Qn))
        return D
    else:
        Ds = []
        for trio in it.combinations(range(len(Qs)),3):
            Jn = np.stack([Js[i] for i in trio],axis=-1)
            Qn = np.stack([Qs[i] for i in trio],axis=-1)
            Ds += [np.einsum('in,nj->ij',Jn,np.linalg.inv(Qn))]
        return Ds

    

