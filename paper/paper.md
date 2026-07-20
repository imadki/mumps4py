---
title: 'mumps4py: A complete Python interface for the MUMPS sparse direct solver'
tags:
  - Python
  - MUMPS
  - sparse linear algebra
  - direct solvers
  - high-performance computing
  - MPI
  - Cython
authors:
  - name: Imad Kissami
    orcid: 0000-0001-6740-1085
    affiliation: 1
  - name: William Piat
    orcid: 0000-0002-1340-2047
    affiliation: 2
affiliations:
  - name: Mohammed VI Polytechnic University, Morocco
    index: 1
  - name: SafranTech, Paris, France
    index: 2
date: 20 July 2026
bibliography: paper.bib
---

# Summary

`mumps4py` is a Python interface to MUMPS (MUltifrontal Massively Parallel
Sparse direct Solver) [@MUMPS:2001; @MUMPS:2006], a widely used library for the
direct solution of large sparse systems of linear equations $Ax = b$. Direct
solvers of this kind are a core building block of many scientific computing
workflows, including finite-element and finite-volume discretizations of partial
differential equations, structural mechanics, computational fluid dynamics and
electromagnetics.

`mumps4py` exposes the full MUMPS C API to Python through a thin, low-overhead
Cython layer, while integrating natively with the standard scientific Python
stack: matrices and vectors are passed as NumPy arrays [@harris2020array] and
distributed-memory parallelism is expressed through `mpi4py` [@dalcin2011parallel].
The package supports **all four arithmetic precisions offered by MUMPS** — real
single (`s`), real double (`d`), complex single (`c`) and complex double (`z`) —
selected at runtime, and covers both centralized and fully distributed assembly
of the matrix, the right-hand side and the solution, as well as elemental input,
Schur complement computation and sparse right-hand sides.

`mumps4py` is listed among the community contributions on the official MUMPS
website [@mumps_website], reflecting its recognition by the upstream project.

# Statement of need

MUMPS is one of the reference sparse direct solvers in high-performance
computing, but it is written in Fortran/C and its primary interface is oriented
toward compiled codes. Python has become a dominant language for prototyping and
running scientific applications, so a robust, fully featured bridge between the
two is valuable to a broad user base.

Existing Python bindings for MUMPS tend to expose only a subset of the solver:
they are commonly restricted to the real double-precision (`d`) driver, to
centralized (single-process) assembly, or to serial usage, which limits their
applicability to large-scale distributed problems and to complex-valued systems
that arise, for example, in frequency-domain wave propagation and quantum
mechanics.

`mumps4py` addresses these limitations by providing:

- **Complete precision coverage.** All four MUMPS arithmetics (`s`, `d`, `c`,
  `z`) are wrapped and can be selected at runtime, so the same interface serves
  real and complex problems in single or double precision.
- **True distributed-memory parallelism.** The interface builds directly on
  `mpi4py` and supports distributed assembly of the matrix
  (`set_rcd_distributed`), the right-hand side (`set_rhs_distributed`) and the
  solution, allowing problems to be solved across many MPI ranks without
  gathering data on a single process.
- **Access to the full feature set.** Beyond the standard analyze/factorize/solve
  cycle, the package exposes control parameters (`ICNTL`/`CNTL`), Schur
  complement extraction (`get_schur`, `expand_schur`), sparse right-hand sides,
  elemental matrix input and matrix export.
- **A low-overhead implementation.** The binding is generated with Cython and
  passes NumPy array buffers to MUMPS by pointer, avoiding unnecessary copies.

The package is portable across Linux, macOS and Windows, is tested against all
four MUMPS drivers via a continuous-integration matrix, and is distributed under
the permissive MIT license. Its inclusion in the official MUMPS list of user
contributions indicates existing uptake within the community.

# Example usage

```python
import numpy as np
from mpi4py import MPI
import mumps4py.mumps_solver as mps

# Select the arithmetic at runtime: "single", "double",
# "complex64" or "complex128".
solver = mps.MumpsSolver(system="double")

# Assemble a sparse matrix in COO form and a right-hand side.
solver.set_coo_centralized(A)      # A is a scipy.sparse COO matrix
solver.set_rhs_centralized(b)      # b is a NumPy array

solver.analyze()
solver.factorize()
solver.solve()                     # solution overwrites the RHS on the host
```

The same code runs under `mpirun -n <p> python script.py`, and the distributed
assembly routines allow each rank to contribute only its local portion of the
matrix and right-hand side.

# Acknowledgements

The author thanks the MUMPS development team for the solver and for listing
`mumps4py` among the community contributions.

# References
