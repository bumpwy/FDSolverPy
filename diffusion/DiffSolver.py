# WKC.py 
# WKC is the class that evolves the WKC 
# phase-field equations. 
#############################################################
import numpy as np
import opt_einsum as oe
import os, subprocess, pickle, gc
import sys
sys.path.append('../')
from base.ParallelSolver import *
from datetime import datetime
import itertools as it


class diff_solver(parallel_solver):
    def __init__(self,
                 # inputs for grid
                 Xs,ghost,
                 # elasticity
                 moduli_object=None,
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
        # setup variables
        self.d = np.zeros(tuple(self.nes+[self.ndim,self.ndim]),dtype=Data_Type[0])
        self.c = np.zeros(tuple(self.nes),dtype=Data_Type[0])
        self.set_variables(varnames=['c'],dat=[self.c],dat_bc=[None],dat_type=[MPI.DOUBLE])
        
        # distribute large grid to decomposed grid (for each cpu)
        if type(D) is str: self.distribute(self.d,np.load(D)['D'])
        else: self.distribute(self.d,D)
        if type(C) is str: self.distribute(self.c,np.load(C)['C'])
        else: self.distribute(self.c,C)

        # for speed
        ops = self.d,tuple(self.nes+[self.ndim])
        self.J_oe_expr = oe.contract_expression('abcij,abcj->abci', *ops, constants=[0])
        self.e_oe_expr = oe.contract_expression('abci,abci->abc',\
                                                tuple(self.nes+[self.ndim]),\
                                                tuple(self.nes+[self.ndim]))

        # parameters
        self.C = C
        # crystal related / moduli tensor
        self.mod = moduli_object
        # store parameters
        self.dict = {'Xs':Xs,'ghost':ghost,'pbc':pbc,
                     'moduli_object':moduli_object,
                     'D':D, 'C':C, 'Data_Type':Data_Type}
    def run(self,outdir='data',restart=False,
            Nstep=100,step=1,etol=1e-4,ftol=1e-4):
        ########## Setup parallel IO ##########
        Nstep,step = int(Nstep), int(step)
        self.initialize_run(outdir,restart=restart)
        Nstep = int(Nstep/step)*step + 1

        # store the class object as dict
        if self.rank == 0:
            pickle.dump(self.dict,
                        open(os.path.join(outdir,'diff_solver.pckl'),'wb'),
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
                       %('it(#)'.ljust(10),'F(eV)'.ljust(15),'dF(eV)'.ljust(15),\
                         'Force_max(eV/A)'.ljust(20),'Time(h:m:s)'.ljust(15)))
        it = 0 # iterator counter
        while (Err > ftol or DF > etol) and it < Nstep:
            # - MPI synchronization
            self.comm.Barrier()
            gc.collect()
            
            # - output data
            if it%step==0:
                self.dump(outdir)
                self.parprint("%s%s%s%s%s"\
                                %(('%i'%it).ljust(10),\
                                  ('%.4e'%Fe0).ljust(15),\
                                  ('%.4e'%(Fe0-Fe_old)).ljust(15),\
                                  ('%.4e'%Err).ljust(20),\
                                  str(datetime.now()-t1).ljust(15)))
            # - integration
            Fe_old = Fe0
            Fe0,Err = self.cg_integrate(self.c,hh,g0,g1)

            # - record results
            if Fe_old==0: DF=0
            else:DF = np.absolute((Fe0-Fe_old)/Fe_old)
            it += 1
        # final output
        self.dump(outdir)
        self.parprint("%s%s%s%s%s"\
                        %(('%i'%it).ljust(10),\
                          ('%.4e'%Fe0).ljust(15),\
                          ('%.4e'%(Fe0-Fe_old)).ljust(15),\
                          ('%.4e'%Err).ljust(20),\
                          str(datetime.now()-t1).ljust(15)))
        self.parprint('Big Loop time lapse: %s'%(str(datetime.now()-t1)))
        ############### The Big Loop ###############

    # relaxation algorithms
    def cg_integrate(self,d,hh,g0,g1,ls_func=None,ls_args={}):
       
        ###### The Line Search #####
        t = self.golden_line_search(hh,d,**ls_args)
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

    def golden_line_search(self,dF_d,d,mu=1e-4,tol_l=1e-1):
        ###### The Golden Line Search #####
        # point 0 ...................
        alpha0 = 0
        Fe0 = self.F(d-alpha0*dF_d)
        # point 1 ...................
        alpha1 = mu
        Fe1 = self.F(d-alpha1*dF_d)
        while Fe1>Fe0:
            alpha1 /= 2
            Fe1 = self.F(d-alpha1*dF_d)
        # point 2 ...................
        alpha2 = 3*alpha1
        Fe2 = self.F(d-alpha2*dF_d)
        while Fe2<Fe1:
            alpha2*=2
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
        for k in range(n-1):
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
    
    def wolfe_line_search(self,dF_d,Fe0,d,mask=None,t=1.,c1=1e-5,c2=0.01,a=0.,b=100.):
        dF0,dF1 = np.zeros_like(dF_d),np.zeros_like(dF_d)
        Fe0,Err0 = self.dF(d,dF0)
        fp0 = (self.comm.allreduce(sendobj=np.sum(dF_d*dF0),op=MPI.SUM))
        while True:
            Fe1,Err1 = self.dF(d-t*dF_d,dF1)
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
    def SecondDiff(self,A,axis):
        # first take gradient
        diffA = np.gradient(A,self.dxs[axis],axis=axis,edge_order=1)
        A_view = np.swapaxes(A,axis,0)
        diffA_view = np.swapaxes(diffA,axis,0)
        # handle edge terms
        diffA_view[0,...] = (A_view[0,...]+0.5*A_view[1,...])/self.dxs[axis]
        diffA_view[-1,...] = -(A_view[-1,...]+0.5*A_view[-2,...])/self.dxs[axis]
        diffA_view[1,...] = (0.5*A_view[2,...]-A_view[0,...])/self.dxs[axis]
        diffA_view[-2,...] = -(0.5*A_view[-3,...]-A_view[-1,...])/self.dxs[axis]

        return diffA
 
    def F(self,c):
        # calculate energy
        self.update_boundary(c)
        gradCs = np.stack(np.gradient(c,*self.dxs),axis=-1)

        ##### NumPy #####
        J = self.J_oe_expr(gradCs)
        e_density = 0.5*self.e_oe_expr(J,gradCs)
        
        
        return self.par_sum((e_density[self.ind]))

    def dF(self,c,dF_dc,mask=None):
        # calculate energy
        self.update_boundary(c)
        gradCs = np.stack(np.gradient(c,*self.dxs),axis=-1)
       
        ##### NumPy approach #####
        J = self.J_oe_expr(gradCs)
        e_density = 0.5*self.e_oe_expr(J,gradCs)
       
        # calculate force
        dF_dc[:] = sum([-self.SecondDiff(J[...,i],i) for i in range(self.ndim)])

        # here we apply a fix boundary condition
        self.fix_boundary(dF_dc)
        self.update_boundary(dF_dc)
        
        # maximal force
        e = np.sqrt(dF_dc**2).max()

        return self.par_sum((e_density[self.ind])),\
               self.comm.allreduce(sendobj=e,op=MPI.MAX)
    def fix_boundary(self,dF_dc):
        for i in range(self.ndim):
            df_dc = np.swapaxes(dF_dc,i,0)
            df_dc[0,...],df_dc[-1,...] = 0, 0



##### helper functions #####
def read_diffsolver_args(fname='diff_solver.pckl'):
    # read in dictionary object
    dct = pickle.load(open(fname,'rb'))
    return dct

def read_diffsolver_data(path):
    # read in dictionary object
    dct = read_diffsolver_args(os.path.join(path,'diff_solver.pckl'))
    Ns = [len(X) for X in dct['Xs']]
    
    # optimized concentration field
    counter_mm = np.memmap(os.path.join(path,'counter.dat'),\
                           dtype='int32',mode='r+',shape=())
    count = counter_mm.tolist()
    C = np.fromfile(os.path.join(path,f'c.{count-1}'),dtype=np.double).reshape(Ns,order='F')

    return dct, C
    
def read_diffsolver(path='./data'):
    # read in data
    dct, C = read_diffsolver_data(path)

    # diffsolver object
    calc = diff_solver(**dct)

    # calculate current
    gradCs = np.stack(np.gradient(C,*calc.dxs),axis=-1)
    J = -np.einsum('abcij,abcj->abci',calc.d,gradCs)

    return calc, C, gradCs, J

def calc_Dijs(dct):
    D = []
    for dat_triplet in it.combinations(dct.values(),3):
        Jms = np.array([dat['Jm'] for dat in dat_triplet])
        gCms = np.array([dat['gCm'] for dat in dat_triplet])
        D += [-np.einsum('im,mj',Jms.T,np.linalg.inv(gCms).T)]
    return D
