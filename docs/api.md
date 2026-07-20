# mumps4py API Reference

This page documents the public API of the `MumpsSolver` class, the single entry
point of `mumps4py`. All index-based MUMPS arguments (`ICNTL`, `CNTL`, row/column
indices, Schur DOFs) follow the **1-based** convention of the MUMPS
documentation.

```python
import mumps4py.mumps_solver as mps
solver = mps.MumpsSolver(system="double")
```

## Typical workflow

```python
solver.set_coo_centralized(A)     # assemble the matrix
solver.set_rhs_centralized(b)     # set the right-hand side
solver.analyze()                  # symbolic analysis (JOB = 1)
solver.factorize()                # numerical factorization (JOB = 2)
solver.solve()                    # solve, RHS is overwritten (JOB = 3)
```

---

## Construction

### `MumpsSolver(system='double', sym=0, par=1, verbose=None, mpi_comm=None, mem_relax=20)`

Initialize the MUMPS solver.

| Parameter | Description |
|-----------|-------------|
| `system`  | Arithmetic/precision: `"single"` (`s`), `"double"` (`d`), `"complex64"` (`c`), `"complex128"` (`z`). |
| `sym`     | Matrix symmetry: `0` = unsymmetric, `1` = symmetric positive definite, `2` = general symmetric. |
| `par`     | Parallel mode: `1` = host takes part in the computation, `0` = host acts only as coordinator. |
| `verbose` | If truthy, enable MUMPS diagnostic output (otherwise the solver is silent). |
| `mpi_comm`| An `mpi4py` communicator. Defaults to `MPI.COMM_WORLD`. |
| `mem_relax` | Memory relaxation percentage (`ICNTL(14)`). |

The chosen `system` must match a MUMPS driver that was compiled in (see
`MUMPS_SOLVERS`); an unsupported value raises `ValueError`.

---

## Control parameters

### `set_icntl(idx, val)`
Set integer control parameter `ICNTL(idx)`. `idx` is 1-based.

### `set_cntl(idx, val)`
Set real control parameter `CNTL(idx)`. `idx` is 1-based.

### `set_verbose()`
Enable error, diagnostic and global-information messages.

### `set_silent()`
Suppress all MUMPS output (the default after construction).

---

## Matrix input — centralized (assembled on the host)

### `set_coo_centralized(mtx)`
Set the sparse matrix from a SciPy sparse matrix in **COO** format. For
symmetric matrices (`sym > 0`) only the upper triangle is kept. Requires a
square matrix.

### `set_rcd_centralized(ir, ic, data, n)`
Set the matrix from explicit **r**ow indices, **c**olumn indices and **d**ata
arrays. `n` is the matrix order. Indices are 1-based; all three arrays must have
the same length.

### `set_data_centralized(data, n)`
Update only the numerical values of an already-structured matrix (same sparsity
pattern), avoiding re-sending the indices.

---

## Matrix input — distributed (each rank contributes its part)

Distributed assembled input requires `ICNTL(18) != 0`.

### `set_rc_distributed(ir_loc, ic_loc, n)`
Set the local row/column indices on each rank. `n` (the global order) is read on
rank 0.

### `set_rcd_distributed(ir_loc, ic_loc, data_loc, n)`
Set the local row indices, column indices and values on each rank.

### `set_data_distributed(data_loc, n)`
Update only the local numerical values (unchanged distributed pattern).

---

## Matrix input — elemental

### `set_elemental_matrix(eltptr, eltvar, a_elt, n, nelt)`
Set the matrix in **elemental** format. Requires `ICNTL(5) = 1` and
`ICNTL(18) = 0`.

| Parameter | Description |
|-----------|-------------|
| `n`       | Order of the matrix. |
| `nelt`    | Number of elements. |
| `eltptr`  | Pointer to the first variable of each element (size `nelt + 1`). |
| `eltvar`  | List of element variables. |
| `a_elt`   | Element values stored by column. |

---

## Right-hand side

### `set_rhs_centralized(rhs)`
Set a dense right-hand side on the host. Accepts a single vector of length `n`
or a 2-D array of several RHS vectors. The solution overwrites this array after
`solve()`.

### `set_rhs_distributed(rhs_loc, irhs_loc, nrhs)`
Set a distributed dense RHS. `irhs_loc` maps local rows to global 1-based
indices. Requires `ICNTL(20)` in `{10, 11}`.

### `set_rhs_sparse(rhs_sparse, rhs_row_indices, rhs_col_ptr, nrhs)`
Set a **sparse** RHS in compressed-column form. `rhs_col_ptr` has size
`nrhs + 1`. Requires `ICNTL(20)` in `{1, 2, 3}`.

### `enable_distributed_solution(nrhs)`
Configure MUMPS to keep the solution distributed across ranks
(`ICNTL(21) = 1`). Must be called after factorization so that `INFO(23)` (the
local solution size) is available.

---

## Solution phases

### `analyze()`
Symbolic analysis (`JOB = 1`).

### `factorize()`
Numerical LU or Cholesky factorization (`JOB = 2`).

### `solve()`
Solve `Ax = b` (`JOB = 3`). The solution is written into the RHS array.

### `__call__(job)`
Run an arbitrary MUMPS `JOB` directly, e.g. `solver(6)` for a combined
analyze + factorize + solve. Errors reported by MUMPS raise `RuntimeError`.

---

## Schur complement

### `get_schur(schur_list)`
Compute and return the (centralized) Schur complement matrix and the condensed
right-hand side for the DOFs in `schur_list` (1-based).

Returns `(schur_matrix, schur_rhs)`.

### `expand_schur(x2)`
Given the solution `x2` restricted to the Schur variables, expand it into the
full global solution vector and return it.

---

## Utilities

### `write_matrix_to_file(filename)`
Ask MUMPS to dump the matrix and RHS to `filename` during the analysis phase
(`write_problem`). Filenames longer than 1023 characters raise `ValueError`.

### `MumpsSolver.pointer_to_numpy(ptr, dtype, shape)` *(static)*
Convert a raw ctypes pointer address returned by MUMPS into a NumPy array view
of the given `dtype` and `shape`.

---

## Error handling

MUMPS error codes are checked after every call. A negative `INFOG(1)` raises a
`RuntimeError` whose message is produced by
`mumps4py.error_handlings.get_mumps_error_message`. The solver is finalized
automatically (`JOB = -2`) when the object is garbage-collected.
