# FDSolverPy

Parallelized finite-difference module for solving physics problem. Here are some notable feature of this library:
1. Adaptability:
   - all parallel-related functionalities are implemented in the `base/ParallelSolver.py`
   - To solve for a new system, the user can inherit the `parallel_solver` class and define the energy function `F` and gradient `dF`
2. Numerical stability:
   - In most optimization codes, the loss function (or the "energy functional") are typically calculated with a simple `sum` of all errors, which is known to be numerically unstable, negatively impacting the convergence speed.
   - While more accurate summation functions are available, e.g. `math.fsum`, they only work on a single-process. An alternative is to use `MPI.allgather` to first gather all arrays to one process, and then perform `fsum` to get accurate results. However, this slows down the process significantly.
   - In the `parallel_solver` class, the `par_sum` member function offers a parallelized and stable summation function to calculate the loss function, without sacrificing speed. This allows for extremely accurate calculation of the loss function, and greatly improves convergence speed for optimization.
   - Implementation details can be found in the `parallel_solver.par_sum` function and also in `math/util.py` (a compensated summation, a.k.a. the Neumaien algorithm, was implemented).


## Installation

The installation time is typically within 5 minutes on a normal local machine.

Dependencies:
- `numpy`
- `mpi4py`
- `numba`
- `matplotlib`

An example for the installation process:

To install `FDSolverPy`, clone this repo:
```bash
git remote add origin git@github.com:bumpwy/FDSolverPy.git
```
and run:
```bash
pip install -e /path/to/the/repo
```

The `-e` option allows you to edit the source code without having to re-install.

To uninstall:
```bash
pip uninstall FDSolverPy
```

## Test run
Examples can be found in the `./test` directory. 
For instance, under `./tests/local/2d/microstruct` contains examples for diffusion problem in a microstructure.
```bash
./run.py
```
and to plot the results do
```
./plot.py
```
and you shall obtain
![alt text](./tests/local/2d/microstruct/results.png)

where $d(\bf{r})$ is the polycrystalline diffusivity, $\Delta c$'s are the perturbative concentration fields for 
driving force in $Q_0 = (1,0)$ and $Q_1 = (0,1)$ directions.
