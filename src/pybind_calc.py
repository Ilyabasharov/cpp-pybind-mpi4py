#!/usr/bin/env python3

import os
import mpi4py
mpi4py.rc.threaded = True
mpi4py.rc.thread_level = "funneled"
from mpi4py import MPI
from mpi_lib import Distributed

os.makedirs('vis/', exist_ok=True)

# Main program
def  main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    distrib = Distributed()
    
    distrib.shoot(100000)
    distrib.seidel(1000, 0)
    
    MPI.Finalize()

if  __name__  == "__main__":
    main()
