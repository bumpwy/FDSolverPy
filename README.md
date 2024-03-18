# FDSolverPy

Parallelized finite-difference module for solving physics problem.


## Installation

The installation time is typically within 5 minutes on a normal local machine.

Dependencies:
- `numpy`
- `mpi4py`
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

where $d(\bold{r})$ is the polycrystalline diffusivity, $\Delta c$'s are the perturbative concentration fields for 
driving force in $Q_0 = (1,0)$ and $Q_1 = (0,1)$ directions.
