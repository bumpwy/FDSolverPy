# WKC.py 
# WKC is the class that evolves the WKC 
# phase-field equations. 
#############################################################
import numpy as np
import opt_einsum as oe
import os, subprocess, pickle, gc, sys, json
from FDSolverPy.base.ParallelSolver import *
from FDSolverPy.math.space import *
from datetime import datetime
import itertools as it



class diff_solver(parallel_solver):
    def __init__(self,
                 # inputs for grid
                 GD,ghost=2,
                 # pbc
                 pbc=0,
                 # diffusivity
                 D='D.npz',
                 # variable initialization
                 C='C.npz',Data_Type=['double'],
                 # extra (mostly for backward compatability)
                 **kwargs):
        # call parent constructor
        parallel_solver.__init__(self,GD,ghost,pbc=pbc)
        
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
        self.d = np.zeros(tuple(self.nes+[self.GD.ndim,self.GD.ndim]),dtype=D_array.dtype)
        self.d_fac = 1
        self.c = np.zeros(tuple(self.nes),dtype=C_array.dtype)
        c_mpi_type = MPI._typedict[self.c.dtype.char]
        self.set_variables(varnames=['c'],dat=[self.c],\
                           dat_bc=[None],dat_type=[c_mpi_type])
        
        # distribute large grid to decomposed grid (for each cpu)
        self.distribute(self.d,D_array)
        self.distribute(self.c,C_array)

        # for speed
        ndim_indices = 'abc'
        ndim_index = ndim_indices[:self.GD.ndim]
        ops = self.d,tuple(self.nes+[self.GD.ndim])
        self.J_oe_expr = oe.contract_expression(f'...ij,...j->...i', *ops, constants=[0])
        self.e_oe_expr = oe.contract_expression(f'...i,...i->...',\
                                                tuple(self.nes+[self.GD.ndim]),\
                                                tuple(self.nes+[self.GD.ndim]))

        # parameters
        self.C = C
        # store parameters
        self.dict = {'GD':GD,'ghost':ghost,'pbc':pbc,
                     'D':D, 'C':C, 'Data_Type':Data_Type}
    def run(self,outdir='data',restart=False,
            Nstep=500,step=20,clean_old=True,
            etol=1e-4,ftol=1e-2,ls_args={"t0":1e-2,"tol":1e-5}):
        
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
            self.dump(outdir,counter,clean_old) # store initial frame if it's a fresh start

        # print run parameters
        self.parprint('RUN PARAMETERS:\n')
        if restart:
            self.parprint('    restart calculation\n')
        else:
            self.parprint('    start from scratch\n')
        self.parprint(f'    etol: {etol:.4e}  ftol: {ftol:.4e}\n')
        self.parprint(f'    line search args: {ls_args}\n')
        self.parprint(f'    Nstep: {Nstep}  step: {step}  clean_old: {clean_old}\n')
        ########## Optimization Setup ##########
        #self.str_to_alg(alg)
        hh,g0,g1 = [np.zeros_like(self.c) for i in range(3)]
        Fe0,Err = self.dF(self.c,g0)
        hh[:] = g0
        Fe_old = 0
        DF = np.Inf
        Fs = [] if restart else [Fe0]
        ############### The Big Loop ###############
        t1 = datetime.now()
        self.parprint('Big Loop Begins...')
        self.parprint("%s%s%s%s"\
                       %('it(#)'.ljust(10),'F(eV)'.ljust(25),\
                         'Force_max(eV/A)'.ljust(20),'Time(h:m:s)'.ljust(15)))
        self.parprint("%s%s%s%s"\
                        %(('%i'%counter).ljust(10),\
                          ('%.12e'%Fe0).ljust(25),\
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
            Fs.append(Fe0)
            if Fe_old==0: DF=0
            else:DF = abs((Fe0-Fe_old)/Fe_old)
            counter += 1
            
            # - output data
            if counter%step==0:
                self.dump(outdir,counter,clean_old)
                self.parprint("%s%s%s%s"\
                                %(('%i'%counter).ljust(10),\
                                  ('%.12e'%Fe0).ljust(25),\
                                  ('%.4e'%Err).ljust(20),\
                                  str(datetime.now()-t1).ljust(15)))
        # final output
        self.parprint('-'*70)
        self.dump_macro_vars(outdir,restart=restart,\
                             Fs=[str(Decimal(F)*Decimal(self.d_fac)) for F in Fs],\
                             #Fs = [str(F) for F in Fs],
                             etol_target=etol,etol_current=str(DF),\
                             ftol_target=ftol,ftol_current=Err,\
                             Nstep=Nstep,counter=counter)
        self.dump(outdir,counter,clean_old)
        self.parprint("%s%s%s%s"\
                        %(('%i'%counter).ljust(10),\
                          ('%.12e'%Fe0).ljust(25),\
                          ('%.4e'%Err).ljust(20),\
                          str(datetime.now()-t1).ljust(15)))
        self.parprint('Big Loop time lapse: %s\n'%(str(datetime.now()-t1)))
        ############### The Big Loop ###############

    # relaxation algorithms
    def cg_integrate(self,d,hh,g0,g1,ls_args={}):
       
        ###### The Line Search #####
        t = self.brent_line_search(d, hh, **ls_args)
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
        if alpha2<1e-6: return 0
        return max(float(alpha1/alpha2), 0)

    def bracket(self,ta,tb,x,d):
        gold,glim,eps = (1+np.sqrt(5))/2, 100, 1e-20
        Fa,Fb = self.F(x-ta*d), self.F(x-tb*d)
        
        # what to do if not downhill? (tb too big)
        if Fb > Fa:
            tb /= gold
            Fb = self.F(x-tb*d)
            #ta, tb = tb, ta
            #Fa, Fb = Fb, Fa
        tc = tb + gold*(tb-ta)
        Fc = self.F(x-tc*d)

        # iteratively determine tc
        while Fb > Fc:
            r, q = (tb-ta)*float(Fb-Fc), (tb-tc)*float(Fb-Fa)
            sgn_qmr = np.sign(q-r) if np.sign(q-r) !=0 else 1
            tu = tb-((tb-tc)*q - (tb-ta)*r)\
                     /(2*sgn_qmr * np.absolute(max(np.absolute(q-r),eps)))
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
    
    def brent_line_search(self,x,d,t0=1e-5,tol=1e-4,maxiter=100):

        # bracketing
        ts,Fs = self.bracket(0,t0,x,d)
        #self.parprint(ts,Fs)
        
        # initialize points
        a, b, c = ts
        Fa, Fb, Fc = Fs
        o, w, v = b, b, b
        Fw = Fv = Fo = self.F(x-o*d)
        e, g = 0, 0
        
        # consts
        gold, eps = 1/(np.sqrt(5)+1), 1e-40

        for i in range(maxiter):
            m = (a + b)*0.5
            tol1 = tol*np.absolute(o)+eps
            tol2 = 2*tol1
            if np.absolute(o-m) <= (tol2-0.5*(b-a)):
                Fmin = Fo
                #self.parprint(o)
                return o
            if (np.absolute(e) > tol1):
                r, q = (o-w)*float(Fo-Fv),(o-v)*float(Fo-Fw)
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

    ##### energy & force #####
    def FirstDiff(self,A,axis):
        # first take gradient
        diffA = np.gradient(A,self.GD.dxs[axis],axis=axis,edge_order=1)
        A_view = np.swapaxes(A,axis,0)
        diffA_view = np.swapaxes(diffA,axis,0)
        
        # handle edge terms
        diffA_view[0,...]  /= np.sqrt(2)
        diffA_view[-1,...] /= np.sqrt(2)

        return diffA

    def SecondDiff(self,A,axis):
        # first take gradient
        diffA = -np.gradient(A,self.GD.dxs[axis],axis=axis)
        A_view = np.swapaxes(A,axis,0)
        diffA_view = np.swapaxes(diffA,axis,0)
        # handle edge terms
        diffA_view[0,...] =  (- 1.0*A_view[0,...]  - 0.5*A_view[1,...])/self.GD.dxs[axis]
        diffA_view[1,...] =  (+ 1.0*A_view[0,...]  - 0.5*A_view[2,...])/self.GD.dxs[axis]
        diffA_view[-2,...] = (+ 0.5*A_view[-3,...] - 1.0*A_view[-1,...])/self.GD.dxs[axis]
        diffA_view[-1,...] = (+ 0.5*A_view[-2,...] + 1.0*A_view[-1,...])/self.GD.dxs[axis]

        return diffA
 
    def F(self,c):
        # calculate energy
        self.update_boundary(c)
        gradCs = np.stack(np.gradient(c,*self.GD.dxs),axis=-1)
        if self.GD.ndim==1: gradCs = np.expand_dims(gradCs,1)

        ##### NumPy #####
        J = self.J_oe_expr(-gradCs)
        e_density = 0.5*self.e_oe_expr(J,-gradCs)*self.GD.dv
         
        return self.par_sum((e_density[self.ind]))

    def dF(self,c,dF_dc,mask=None):
        # calculate energy
        self.update_boundary(c)
        gradCs = np.stack(np.gradient(c,*self.GD.dxs),axis=-1)
        if self.GD.ndim==1: gradCs = np.expand_dims(gradCs,1)
       
        ##### NumPy approach #####
        J = self.J_oe_expr(-gradCs)
        e_density = 0.5*self.e_oe_expr(J,-gradCs)*self.GD.dv
       
        # calculate force
        dF_dc[:] = sum([self.SecondDiff(-J[...,i],i) for i in range(self.GD.ndim)])

        # here we apply a fix boundary condition
        self.fix_boundary(dF_dc)
        self.update_boundary(dF_dc)
        
        # maximal force
        e = np.sqrt(dF_dc[self.ind]**2).max()
         
        return self.par_sum((e_density[self.ind])),\
               self.comm.allreduce(sendobj=e,op=MPI.MAX)
    def fix_boundary(self,dF_dc):
        for i in range(self.GD.ndim):
            if self.pbc[i]: continue
            df_dc = np.swapaxes(dF_dc,i,0)
            df_dc[0,...],df_dc[-1,...] = 0, 0

    # storing macro variables
    def dump_macro_vars(self,outdir,restart,**kwargs):

        # macro Q and J
        self.update_boundary(self.c)
        q = -np.stack(np.gradient(self.c,*self.GD.dxs),axis=-1)
        if self.GD.ndim==1: q = np.expand_dims(q,axis=1)
        j = self.J_oe_expr(q)*self.d_fac
        Q,J = (self.par_mean(q[self.ind])), self.par_mean(j[self.ind])

        # D_par and D_ser
        D_par = self.par_mean((self.d[self.ind]*self.d_fac))
        s, logdet_d = np.linalg.slogdet(self.d[self.ind]*self.d_fac)
        if 0 not in s: 
            d_inv = np.linalg.inv(self.d)
            D_ser = np.linalg.inv(self.par_mean(d_inv))*self.d_fac
        else:
            D_ser = np.zeros((self.GD.ndim,self.GD.ndim))

        # storage
        if self.rank==0:
            fname = os.path.join(outdir,'macro_vars.json')
            if restart:
                try:
                    with open(fname,'rt') as f:
                        data = json.load(f)
                except FileNotFoundError:
                    data = {}
            else:
                data = {}
            # merge old and new data
            for key in kwargs:
                if key in data and type(data[key])==list:
                    data[key] += kwargs[key]
                else:
                    data[key] = kwargs[key]

            data.update({'Q':Q.tolist(),'J':J.tolist(),'D_par':D_par.tolist(),'D_ser':D_ser.tolist()})
            json.dump(data,
                      open(fname,'w'),indent=4)

##### pbc diffsolver class #####
class diff_solver_pbc(diff_solver):
    def __init__(self,
                 # inputs for grid
                 GD,ghost=2,
                 # diffusivity
                 D='D.npz',Q=[1,0,0],
                 # variable initialization
                 C='C.npz',Data_Type=['double'],
                 # extra (mostly for backward compatability)
                 **kwargs):
        # call parent constructor
        diff_solver.__init__(self,GD,ghost,pbc=1,D=D,C=C,Data_Type=Data_Type,**kwargs)

        # additional parameters
        self.Q = np.array(Q)

    # overload Energy/Force functions for pbc case
    def F(self,c):
        # calculate energy
        self.update_boundary(c)
        dqs = -np.stack(np.gradient(c,*self.GD.dxs),axis=-1)

        ##### NumPy #####
        J = self.J_oe_expr(dqs+self.Q)
        e_density = 0.5*self.e_oe_expr(J,dqs+self.Q)
         
        return self.par_sum((e_density[self.ind]))

    def dF(self,c,dF_dc,mask=None):
        # calculate energy
        self.update_boundary(c)
        dqs = -np.stack(np.gradient(c,*self.GD.dxs),axis=-1)
       
        ##### NumPy approach #####
        J = self.J_oe_expr(dqs+self.Q)
        e_density = 0.5*self.e_oe_expr(J,dqs+self.Q)
       
        # calculate force
        dF_dc[:] = sum([np.gradient(J[...,i],dx,axis=i) for i,dx in enumerate(self.GD.dxs)])

        # here we apply boundary condition
        self.update_boundary(dF_dc)
        
        # maximal force
        e = np.sqrt(dF_dc**2).max()
         
        return self.par_sum((e_density[self.ind])),\
               self.comm.allreduce(sendobj=e,op=MPI.MAX)
    # storing macro variables
    def dump_macro_vars(self,outdir,**kwargs):

        # macro Q and J
        self.update_boundary(self.c)
        dq = -np.stack(np.gradient(self.c,*self.GD.dxs),axis=-1)
        j = self.J_oe_expr(dq+self.Q)*self.d_fac
        J = (self.par_mean(j))

        # D_par and D_ser
        D_par = (self.par_mean(self.d))*self.d_fac
        d_inv = np.linalg.inv(self.d)
        D_ser = np.linalg.inv((self.par_mean(d_inv)))*self.d_fac

        # storage
        if self.rank==0:
            fname = os.path.join(outdir,'macro_vars.json')
            kwargs.update({'Q':self.Q.tolist(),'J':J.tolist(),'D_par':D_par.tolist(),'D_ser':D_ser.tolist()})
            json.dump(kwargs,
                      open(fname,'w'),indent=4)
##### pbc diffsolver class #####

##### helper functions #####
def normalize_parameters(calc,nn_value=None):
    # normalize paramters before calculation
    # greatly enhances stability for small d's
    #Tr_d = np.diagonal(calc.d[calc.ind],axis1=-2,axis2=-1).mean(axis=-1)
    Tr_d = np.diagonal(calc.d,axis1=-2,axis2=-1).mean(axis=-1)
    if not nn_value:
        nn_value = float(calc.par_mean(Tr_d[calc.ind]))
    calc.d/=nn_value
    calc.d_fac = nn_value
    _,F_max = calc.dF(calc.c,np.zeros_like(calc.c))

    calc.parprint(f'normalized diffusivity d by {nn_value}, with F_max:{F_max}\n')

    return nn_value, F_max

def read_diffsolver_args(path='.'):
    # read in dictionary object
    dct = pickle.load(open(os.path.join(path,'diff_solver.pckl'),'rb'))
    return dct

def read_diffsolver_data(path='.',prefix='data',frame=-1):
    # read in dictionary object
    dct = read_diffsolver_args(path)

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
    
    buff = open(os.path.join(data_path,f'c.{count}'),'rb').read()
    C_stream = np.frombuffer(buff,dtype=np.uint8)
    C = C_stream.view(dtype=np.double).reshape(dct['GD'].ns,order='F')


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
    calc.distribute(calc.c,c)
    q = -np.stack(np.gradient(calc.c,*calc.GD.dxs),axis=-1)
    if calc.GD.ndim==1: q = np.expand_dims(q,axis=1)
    j = np.einsum(f'...ij,...j->...i',calc.d,q)

    return calc, c, q[calc.ind], j[calc.ind]

# write inputs
def write_inputs(path,input_dct,C,D,compressed=True):
    # create path (if not exist)
    subprocess.call(f'mkdir -p {path}',shell=True)
    print(f'Created input files at {path}')
    
    #- input args file
    pickle.dump(input_dct,\
                open(os.path.join(path,'diff_solver.pckl'),'wb'),\
                protocol=-1)

    #- C file
    if isinstance(C,str):
        subprocess.call(f'cp {C} {os.path.join(path,"C.npz")}',shell=True)
    else:
        if compressed:
            np.savez_compressed(os.path.join(path,'C.npz'),C=C)
        else:
            np.savez(os.path.join(path,'C.npz'),C=C)
    
    #- D file
    if isinstance(D,str):
        subprocess.call(f'cp {D} {os.path.join(path,"D.npz")}',shell=True)
    else:
        if compressed:
            np.savez_compressed(os.path.join(path,'D.npz'),D=D)
        else:
            np.savez(os.path.join(path,'D.npz'),D=D)

def write_d_eff_inputs(path,D,grid,compressed=True,**kwargs):
    # create path (if not exist)
    subprocess.call(f'mkdir -p {path}',shell=True)

    dim = len(grid.ns)
    Qs = np.eye(dim)
    for i,Q in enumerate(Qs):
    
        # create initial value
        C = macro_C0(grid,Q)

        # input dict
        input_dct = {'GD':grid}
        input_dct.update(kwargs)

        # write inputs
        sub_path = os.path.join(path,f'Q_{i}')  # sub_path
        write_inputs(sub_path,input_dct,C,D,compressed)

def check_d_eff_outputs(path,ftol=None,etol=None):
    q_paths = sorted(glob.glob(os.path.join(path,'Q_*')),key=lambda x: int(x.split('_')[-1]))
    N = len(q_paths)
    completion = 0
    etol_targets, ftol_targets = [], []
    etol_currents, ftol_currents = [], []
    
    for q_path in q_paths:
        out_file = os.path.join(q_path,'data/macro_vars.json')
        if not os.path.isfile(out_file):
            continue
        output_vars = json.load(open(out_file,'r'))
        if etol is None:
            etol_target = output_vars['etol_target']
        if ftol is None:
            ftol_target = output_vars['ftol_target']

        if output_vars['etol_current']<=etol_target and \
            output_vars['ftol_current']<=ftol_target:
            completion += 1
        etol_targets.append(etol_target)
        ftol_targets.append(ftol_target)
        etol_currents.append(output_vars['etol_current'])
        ftol_currents.append(output_vars['ftol_current'])

    return {'N':N,'completion':completion,\
            'etol_targets':etol_targets,'etol_currents':etol_currents,\
            'ftol_targets':ftol_targets,'ftol_currents':ftol_currents}
    

# create the initial concentration field according to Q-vector
# if Q is not provided, i.e. Q=None, a randomized unit vector will be used

def micro_delC(d,grid,Q):
    ndim = len(grid.ns)
    grad_inv_d = grad_fft(1/d,grid)
    indices = 'abc'[:ndim]
    f = np.einsum(f'{indices},{indices}i->{indices}i',d,grad_inv_d)
    fQ = np.einsum(f'{indices}i,i->{indices}',f,Q)

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
    ndim = len(grid.ns)
    indices = 'abc'[:ndim]
    C0 = np.einsum(f'i,i{indices}->{indices}',-Q,np.array(grid.xxs))

    return C0

def create_C(grid,d,Q=None):

    return macro_C0(grid,Q) + micro_delC(d,grid,Q)


# calculates effective diffusivity from given calculation results
# If three are given then spits out one D_ij, but if more are given
# it spits out results of all combinations of 3
def calculate_D(paths=['Q_0','Q_1','Q_2'],prefixes='data',output='D_eff.json'):
    if isinstance(prefixes,str):
        prefixes = [prefixes]*len(paths)

    #- now load all data 
    Qs,Js = [], []
    for path,prefix in zip(paths,prefixes):
        macro_vars_file = os.path.join(path,prefix,'macro_vars.json')
        if os.path.isfile(macro_vars_file):
            dat = json.load(open(macro_vars_file,'r'))
            J,Q,D_par,D_ser = dat['J'], dat['Q'], dat['D_par'], dat['D_ser']
        # the following part doesn't work for pbc case
        else:
            #- load C, Q and update calc
            calc = diff_solver(**read_diffsolver_args())
            dct, C = read_diffsolver_data(path)
            calc.Q = dct['Q']
            calc.distribute(calc.c,C)
            
            #- calculate q & j
            q = -np.stack(np.gradient(calc.c,*calc.GD.dxs),axis=-1)
            j = np.einsum(f'...ij,...j->...i',calc.d,q)
            q,j = q[calc.ind], j[calc.ind]
            
            #- take average 
            axes = tuple(range(calc.GD.ndim))
            J,Q = j.mean(axis=axes), q.mean(axis=axes)
    
        Js += [J]
        Qs += [Q]
    
    # Second, pick 3 out of all results and calculate effective diffusivity
    n = len(paths)
    ndim = len(Js[0])
    if n==ndim:
        Jn = np.stack(Js,axis=-1)
        Qn = np.stack(Qs,axis=-1)
        Ds = np.einsum('in,nj->ij',Jn,np.linalg.inv(Qn))
    else:
        Ds = []
        for trio in it.combinations(range(len(Qs)),ndim):
            Jn = np.stack([Js[i] for i in trio],axis=-1)
            Qn = np.stack([Qs[i] for i in trio],axis=-1)
            Ds += [np.einsum('in,nj->ij',Jn,np.linalg.inv(Qn))]

    output_dict = {'Ds':Ds.tolist(),'D_par':D_par,'D_ser':D_ser}
    json.dump(output_dict,open(output,'w'),indent=4)
    return output_dict
