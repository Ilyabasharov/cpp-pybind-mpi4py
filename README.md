# Effective Numerical Calculus in Python

## Introduction

Demonstrates how to call a C++ class from Python using pybind11 together with MPI.
The task is to solve differential equations using [Shooting](https://en.wikipedia.org/wiki/Shooting_method) and [Seidel](https://www.cfm.brown.edu/people/dobrush/am34/Mathematica/ch4/seidel.html) methods.

<div align="center">

[![license](https://shields.io/badge/license-MIT-green)](https://github.com/Ilyabasharov/cpp-pybind-mpi4py/blob/main/LICENSE)

[📘Introduction](https://github.com/Ilyabasharov/cpp-pybind-mpi4py/blob/main/README.md#introduction) |
[🛠️Installation](https://github.com/Ilyabasharov/cpp-pybind-mpi4py/blob/main/README.md#installation) |
[👀Project Structure](https://github.com/Ilyabasharov/cpp-pybind-mpi4py/blob/main/README.md#contents)

</div>

## Installation

### Requirements

```bash
apt install openmpi
apt install cmake
```

### Project Installation

```bash
git clone --recursive https://github.com/Ilyabasharov/cpp-pybind-mpi4py.git
pip install -r requirements.txt
cd cpp-pybind-mpi4py
mkdir -p build && cd build
export CC=gcc
export CXX=g++
cmake ..
make
```

## Basic run

```bash
mpirun -n 8 python3 helloWorld.py
```


## Contents
 
Main files:

- `mpi_lib.cpp`: C++/MPI library code
- `pybind_calc.py`: python code that call the C++/MPI library (using pybind11 and mpi4py)
