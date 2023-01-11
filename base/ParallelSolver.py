#############################################################
# Base class for a 3D parallel solver, taking care of the 
# parallelization, domain decomposition, communication, 
# MPI-IO, counters, etc.
#############################################################
import os,subprocess, glob
import numpy as np
from mpi4py import MPI
from math import fsum
#import accupy as ap
import mpmath as mp

class parallel_solver():
    def __init__(self,Xs,ghost=1,pbc=(0,0,0),partition=None):

        ########## global grid ##########
        # Grid size
        self.Ns = [len(x) for x in Xs]
        self.Xs = Xs
        self.dxs = [x[1]-x[0] for x in Xs]
        self.ndim = len(Xs)
        self.Buffer = np.empty((self.Ns[0]*self.Ns[1]*self.Ns[2],),dtype=np.double)
        
        ########## local grid ##########
        # domain decomposition
        self.comm,self.partition,self.rank_coord,\
            self.neighbors, self.ns_list, self.disps_list = domain_decomposition(MPI.COMM_WORLD,self.Ns,pbc,partition)
        self.rank,self.comm_size = self.comm.rank, self.comm.size

        ##### testing #####
        coords_list = [self.comm.Get_coords(i) for i in range(self.comm.size)]
        self.sizes_list = []
        for c in coords_list:
            s = 1
            for i in range(self.ndim):
                s *= self.ns_list[i][c[i]]
            self.sizes_list += [s]
        self.d1_disps_list = np.cumsum([0]+self.sizes_list[:-1])
        ##### testing #####
     
        # local grid
        self.ghost = ghost
        self.ns = [n[coord] for n,coord in zip(self.ns_list,self.rank_coord)]
        self.disps = [disp[coord] for disp, coord in zip(self.disps_list,self.rank_coord)]
        self.nes,self.xs,ind = [],[],[]
        for i in range(self.ndim):
            # determine whether to add ghost region
            start,end = 1,1
            if (not pbc[i]) and (self.neighbors[i][0]<0): start = 0
            if (not pbc[i]) and (self.neighbors[i][1]<0): end = 0
            # create local grid
            self.nes += [self.ns[i]+(start+end)*ghost]
            self.xs += [Xs[i][self.disps[i]-start*ghost:self.disps[i]+self.nes[i]]]
            ind += [np.s_[start*ghost:self.nes[i]-end*ghost]]

        self.ind = tuple(ind) 
        self.xxs = np.meshgrid(*self.xs,indexing='ij')
        self.rr = np.stack(tuple(self.xxs),axis=-1)

        # senf/recv buffers for mpi communication, in x, y, z directions 
        self.send_buff, self.recv_buff = [], []
        for i in range(self.ndim):
            nes_swap = self.nes.copy()
            nes_swap[i] = nes_swap[0]
            nes_swap[0] = self.ghost
            self.send_buff += [[np.zeros(tuple(nes_swap),order='F') for i in range(2)]]
            self.recv_buff += [[np.zeros(tuple(nes_swap),order='F') for i in range(2)]]

        ########## output status ##########
        self.parprint("Running on %i cores"%self.comm.size)
        self.parprint("Domain decomposition %ix%ix%i"%tuple(self.partition))
    def set_variables(self,varnames,dat,dat_bc,dat_type):
        # setup data pointers
        self.n_var, self.varnames = len(varnames), varnames
        self.dat, self.dat_bc, self.dat_type = dat, dat_bc, dat_type
        self.dat_r = []
        for i in range(self.n_var):
            if dat_type[i] is MPI.BYTE:
                d_type='i1'
            elif dat_type[i] is MPI.FLOAT:
                d_type='float32'
            else:
                d_type='double'
            self.dat_r.append(np.zeros(tuple(self.ns),order='F',dtype=d_type))
    def run(self,Nstep=1e6,step=1000,filename='data',restart=False):
        # setup parallel IO
        self.initialize_run(filename,int(Nstep),int(step),restart=restart)
        ############### The Big Loop ##############
        t1 = MPI.Wtime()
        for i in range(Nstep):
            if i%step==0:
                self.parprint('step %.i'%i)
        for i in range(self.n_var):
            self.dat_mpi_arraytype[i].Free()
        self.phi_mpifh.Close()
        self.theta_mpifh.Close()
        t2 = MPI.Wtime()
        self.parprint('Big Loop took: %.5f secs'%(t2-t1))
        ############### The Big Loop ###############
    def locate_rank(self,pos=[0,0],index=False):
        if not index:
            i,j = np.digitize(pos[0],self.X)-1,np.digitize(pos[1],self.Y)-1
        else:
            i,j = pos[0],pos[1]
        n,m = np.digitize(i,self.disps_n)-1,np.digitize(j,self.disps_m)-1
        return self.comm.Get_cart_rank([n,m])
    def par_sum(self,a):

        #### method 1: very slow, very accurate ####
        #self.comm.Allgatherv([a.flatten(),MPI.DOUBLE],\
        #                     [self.Buffer,self.sizes_list,self.d1_disps_list,MPI.DOUBLE])
        #return fsum(self.Buffer)

        #### method 2: somewhat faster, quite accurate ####
        #A=np.ravel(self.comm.allgather(fsum(a.ravel())))
        #return fsum(A)

        #### method 3: fast, and somewhat accurate ####
        #A=np.ravel(self.comm.allgather(ap.ksum(a.ravel(),K=2)))
        A=np.ravel(self.comm.allgather(fsum(a.ravel())))
        return fsum(A)
        
    def update_boundary(self,dat,*argv):
        dim = len(dat.shape)-self.ndim
        if dim==0:
            #if len(argv)>0:
            #    self.set_boundary(dat,*argv[0])
            self._update_boundary(dat)
        else:
            grid = (np.s_[...],)
            for i,index in enumerate(np.ndindex(tuple([self.ndim]*dim))):
                self._update_boundary(dat[grid+index])
    def _update_boundary(self,dat):
        for i in range(self.ndim):
            self.reqs = []
            dat_swap = np.swapaxes(dat,i,0)
            # fill in boundary values to send
            if self.neighbors[i][0] >= 0:
                self.send_buff[i][0][:] = dat_swap[self.ghost:2*self.ghost,...]
            if self.neighbors[i][1] >= 0:
                self.send_buff[i][1][:] = dat_swap[-2*self.ghost:-self.ghost,...]
            # MPI requests for sending
            self.reqs.append(self.comm.Isend(self.send_buff[i][0],dest=self.neighbors[i][0],tag=0))
            self.reqs.append(self.comm.Isend(self.send_buff[i][1],dest=self.neighbors[i][1],tag=1))
            # MPI requests for receiving
            self.reqs.append(self.comm.Irecv(self.recv_buff[i][0],source=self.neighbors[i][0],tag=1))
            self.reqs.append(self.comm.Irecv(self.recv_buff[i][1],source=self.neighbors[i][1],tag=0))
            MPI.Request.Waitall(requests=self.reqs)
            # fill in received boundary values
            if self.neighbors[i][0] >= 0: 
                dat_swap[:self.ghost,...] = self.recv_buff[i][0]
            if self.neighbors[i][1] >= 0: 
                dat_swap[-self.ghost:,...] = self.recv_buff[i][1]

    def UpdateAllVariableBoundaries(self):
        for i, vn in enumerate(self.varnames):
            self.update_boundary(self.dat[i],self.dat_bc[i])
    def DistributeAllVariables(self,dat,frame):
        for i, vn in enumerate(self.varnames):
            if vn not in dat.keys():
                continue
            self.distribute(self.dat[i],dat[vn][...,frame])
            #self.update_boundary(self.dat[i],self.dat_bc[i])
    
    def distribute(self,var,Dat):
        dat_rank_inds = tuple(var.shape[self.ndim:])
        # Master process: partition Dat and send out to each processor
        if self.rank==0:
            send_reqs = []
            for p in range(1,self.comm.size):
                # obtain rank-p's coordinate, size, and displacement
                coord = self.comm.Get_coords(p)
                ns_p = [self.ns_list[i][coord[i]] for i in range(self.ndim)]
                disps_p = [self.disps_list[i][coord[i]] for i in range(self.ndim)]

                # populate send-buffer 
                sendbuf = np.empty(tuple(ns_p)+dat_rank_inds,dtype=var.dtype,order='F')
                sendbuf[:] = Dat[tuple([np.s_[disp_p:disp_p+n_p] \
                                        for disp_p,n_p in zip(disps_p,ns_p)]+[np.s_[...]])]
                # send to Worker-p (rank-p)
                send_reqs += [self.comm.Isend(sendbuf,dest=p,tag=p)]
            MPI.Request.Waitall(send_reqs)
            # create Master's recv-buffer (an artificial one of course...)
            recvbuf = \
                    Dat[tuple([np.s_[disp:disp+n] for disp,n in zip(self.disps,self.ns)]+[np.s_[...]])]
        else:
            # create Worker's receive-buffer and receive from Master
            recvbuf = np.empty(tuple(self.ns)+dat_rank_inds,dtype=var.dtype,order='F')
            recv_req = self.comm.Irecv(recvbuf,source=0,tag=self.rank)
            MPI.Request.Wait(recv_req)

        # populate everybody's var-array from recvbuf
        var[self.ind+(np.s_[...],)] = recvbuf
        self.update_boundary(var)


        #var[self.ind+(np.s_[...],)] =\
        #        Dat[tuple([np.s_[disp:disp+n] for disp,n in zip(self.disps,self.ns)]+[np.s_[...]])]
        #self.update_boundary(var)

    def set_boundary(self,var,Left,Right,Bottom,Top,\
                              LowerLeft,LowerRight,UpperLeft,UpperRight):
        # left (x- boundary)
        if self.n_prev<0:
            if isinstance(Left,str) and Left == 'cont':
                for i in range(self.ghost):
                    var[self.ghost-i-1,self.ghost:-self.ghost] = \
                            var[self.ghost+1-i,self.ghost:-self.ghost]
            elif hasattr(Left,'__len__'):
                var[:self.ghost,self.ghost:-self.ghost] = \
                        Left[:,self.disp_m:self.disp_m+self.ny]
            elif Left is None: pass
            else: var[:self.ghost,self.ghost:-self.ghost] = Left
        # right (x+ boundary)
        if self.n_nxt<0:
            if isinstance(Right,str) and Right == 'cont':
                for i in range(self.ghost):
                    var[-self.ghost+i,self.ghost:-self.ghost] = \
                            var[-self.ghost+i-2,self.ghost:-self.ghost]
            elif hasattr(Right,'__len__'):
                var[-self.ghost:,self.ghost:-self.ghost] = \
                        Right[:,self.disp_m:self.disp_m+self.ny]
            elif Right is None: pass
            else: var[-self.ghost:,self.ghost:-self.ghost] = Right
        # bottom (y- boundary)
        if self.m_prev<0:
            if isinstance(Bottom,str) and Bottom == 'cont':
                for i in range(self.ghost):
                    var[self.ghost:-self.ghost,self.ghost-i-1] = \
                            var[self.ghost:-self.ghost,self.ghost+1-i]
            elif hasattr(Bottom,'__len__'):
                var[self.ghost:-self.ghost,:self.ghost] = \
                        Bottom[self.disp_n:self.disp_n+self.nx,:]
            elif Bottom is None: pass
            else: var[self.ghost:-self.ghost,:self.ghost] = Bottom
        # top (y+ boundary)
        if self.m_nxt<0:
            if isinstance(Top,str) and Top == 'cont':
                for i in range(self.ghost):
                    var[self.ghost:-self.ghost,-self.ghost+i] = \
                            var[self.ghost:-self.ghost,-self.ghost+i-2]
            elif hasattr(Top,'__len__'):
                var[self.ghost:-self.ghost,-self.ghost:] = \
                        Top[self.disp_n:self.disp_n+self.nx,:]
            elif Top is None: pass
            else: var[self.ghost:-self.ghost,-self.ghost:] = Top
        ##### Corners #####
        # left-bottom corner
        if self.n_prev<0 and self.m_prev<0:
            if LowerLeft is None: pass
            else: var[:self.ghost,:self.ghost]=LowerLeft
        # right-bottom corner
        if self.n_nxt<0 and self.m_prev<0:
            if LowerRight is None: pass
            else: var[-self.ghost:,:self.ghost] = LowerRight
        # left-top corner
        if self.n_prev<0 and self.m_nxt<0:
            if UpperLeft is None: pass
            else: var[:self.ghost,-self.ghost:] = UpperLeft
        # right-top corner
        if self.n_nxt<0 and self.m_nxt<0:
            if UpperRight is None: pass
            else: var[-self.ghost:, -self.ghost:] = UpperRight

    def initialize_run(self, outdir, restart=False):
        ############################################################
        # initialize_run function makes all the preparations for a 
        # run to start. This includes:
        #       1. MPI-IO file pointers
        #       2. initial values of variables, whether from user given initialized values,
        #          or from the last frame of the previous run if in "restart" mode
        #       3. Also, a counter pointer for the rank=0 process 
        ############################################################
        
        ########## MPI-IO Stuff ##########
        # open outdir
        if self.rank==0:
            data_paths = os.path.join(outdir,'*')
            if glob.glob(data_paths) != []:
                subprocess.call(f'rm {data_paths}',shell=True)
            subprocess.call(f'mkdir -p {outdir}',shell=True)
        # derived array data type for MPI-IO. defines pointer to the sub-block of array for each processor
        self.dat_mpi_arraytype = \
            [d_type.Create_subarray(self.Ns,self.ns,self.disps,order=MPI.ORDER_F) for d_type in self.dat_type]
        [a_type.Commit() for a_type in self.dat_mpi_arraytype]

        # open MPI I/O file handle
        if restart:
            if self.rank==0:
                self.counter_mm = np.memmap(os.path.join(outdir,'counter.dat'),\
                                            dtype='int32',mode='r+',shape=())
                counter = self.counter_mm.tolist()
            else: counter = None
            counter = self.comm.bcast(counter,root=0)
            # Read the last frame from the Previous Run
            amode = MPI.MODE_RDONLY
            for i in range(self.n_var):
                full_fname = os.path.join(path,f'{self.varnames[i]}.{counter}')
                mpifh = MPI.File.Open(self.comm, full_fname, amode)
                mpifh.Set_view(0,etype=self.dat_type[i],filetype=self.dat_mpi_arraytype[i])
                mpifh.Read(self.dat_r[i])
                self.dat[i][self.ind] = self.dat_r[i]
                mpifh.Close()
        else:
            if self.rank==0:
                self.counter_mm = np.memmap(os.path.join(outdir,'counter.dat'),\
                                            dtype='int32',mode='w+',shape=())
        # update boundary
        for dd in self.dat:self.update_boundary(dd)
    
    # dump data to file
    def dump(self,outdir):
        
        # sync counter 
        if self.rank==0: 
            counter=self.counter_mm.tolist()
        else: counter=None
        counter = self.comm.bcast(counter,root=0)
       
        amode = MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_EXCL
        
        for i in range(self.n_var):
            # filename, and also check if exists
            full_fname = os.path.join(outdir,f'{self.varnames[i]}.{counter}')
            if self.rank==0 and os.path.exists(full_fname): 
                subprocess.call(f'rm {full_fname}',shell=True)
            
            # dump data
            self.dat_r[i][:] = self.dat[i][self.ind]
            fh = MPI.File.Open(self.comm,full_fname,amode)
            fh.Set_view(0,etype=self.dat_type[i],filetype=self.dat_mpi_arraytype[i])
            fh.Write_all(self.dat_r[i])
            fh.Close()
            
        # counter increment
        if self.rank==0: 
            self.counter_mm += 1 
            self.counter_mm.flush()
    # parallel printing
    def parprint(self,*argv):
        if self.rank==0: print(*argv)

########## helper functions ##########
def extend_grid(x,ghost,pbc=False):
    if pbc: mode='wrap'
    else: mode='reflect'
    return np.pad(x,[ghost,ghost],mode=mode)

def domain_decomposition(comm,Ns,pbc=(),partition=None):
    ###########################################################
    # determines the most efficient (min perimeter) domain
    # decomposition given the # of cpus (comm.size)
    # - inputs:
    # comm: mpi communication objects
    # Ns: grid size as list [Nx,Ny,Nz]
    # pbc: flag for periodic boundary conditions in each direction
    # partition: user is welcome to provide decomposition for cpus
    # - outputs:
    # cart_comm: the cartesian communicator
    # ns: cpu decomposition as a list [nx,ny,nz]
    # rank_coord: coordinate of current rank in the cartesian comm
    # sizes: sizes of each sub-grid
    # disps: array index for first element for each sub-grid  
    # neighbors: rank number of neighboring processes 
    ###########################################################
    size = comm.size
    ndim = len(Ns)
    if partition==None:
        # find decomposition with minimum perimeter for each block
        # -- 1-dimension case
        if ndim == 1: partition = [size]
        elif ndim >= 2:
            prime_list = prime_sieve(size)
            perimeter = np.Inf
        # -- 2-dimension case
            if ndim==2:
                for n0 in get_factors(size,prime_list):
                    p = Ns[0]//n0 + Ns[1]//(size/n0)
                    if p < perimeter:
                        perimeter = p
                        partition = [n0,int(size/n0)]
        # -- 3-dimension case
            elif ndim==3:
                for n0 in get_factors(size,prime_list):
                    for n1 in get_factors(size/n0,prime_list):
                        l1,l2,l3 = Ns[0]//n0, Ns[1]//n1, Ns[2]//(size/n0/n1)
                        p = (l1*l2 + l2*l3 + l1*l3)*2
                        if p < perimeter:
                            perimeter = p
                            partition = [n0,n1,int(size/n0/n1)]
    # create Cartesian communicator
    cart_comm = comm.Create_cart(dims=partition,periods=pbc,reorder=False)
    # determine location among processors
    neighbors = [cart_comm.Shift(direction=i,disp=1) for i in range(ndim)]
    rank = cart_comm.rank
    rank_coord = cart_comm.Get_coords(rank)
    # determine size/dimension of each sub-grid 
    rems = [N%p for N,p in zip(Ns,partition)]
    ns_list = [[N//p+1]*rem + [N//p]*(p-rem) for N,p,rem in zip(Ns,partition,rems)]
    disps_list = [[(N//p+1)*i for i in range(rem)]+\
                  [(N//p+1)*rem + (N//p)*i for i in range(p-rem)] for (N,p,rem) in zip(Ns,partition,rems)]
    
    return cart_comm, partition, rank_coord, neighbors, ns_list, disps_list
        
# return a dict or a list of primes up to N
# create full prime sieve for N=10^6 in 1 sec
def prime_sieve(n, output={}):
    nroot = int(np.sqrt(n))
    sieve = list(range(n+1))
    sieve[1] = 0

    for i in range(2, nroot+1):
        if sieve[i] != 0:
            m = n//i - i
            sieve[i*i: n+1:i] = [0] * (m+1)

    if type(output) == dict:
        pmap = {}
        for x in sieve:
            if x != 0:
                pmap[x] = True
        return pmap
    elif type(output) == list:
        return [x for x in sieve if x != 0]
    else:
        return None

# get a list of all factors for N
# ex: get_factors(10) -> [1,2,5,10]
def get_factors(n, primelist=None):
    if primelist is None:
        primelist = prime_sieve(n,output=[])

    fcount = {}
    for p in primelist:
        if p > n:
            break
        if n % p == 0:
            fcount[p] = 0

        while n % p == 0:
            n /= p
            fcount[p] += 1

    factors = [1]
    for i in fcount:
        level = []
        exp = [i**(x+1) for x in range(fcount[i])]
        for j in exp:
            level.extend([j*x for x in factors])
        factors.extend(level)
    return factors
