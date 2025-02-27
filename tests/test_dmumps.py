import numpy as np
import pytest
import os
from mumps4py.mumps_solver import MumpsSolver

@pytest.mark.skipif("dmumps" not in os.getenv("MUMPS_SOLVERS", "").split(","), reason="dmumps not selected")
def test_solve_double():
    solver = MumpsSolver(verbose=False, system="double")

    dtype = np.float64
    n = 5
    irn = np.array([0,1,3,4,1,0,4,2,1,2,0,2], dtype=np.int32)
    jcn = np.array([1,2,2,4,0,0,1,3,4,1,2,2], dtype=np.int32)
    a = np.array([3.0,-3.0,2.0,1.0,3.0,2.0,4.0,2.0,6.0,-1.0,4.0,1.0], dtype=dtype)
    b = np.array([20.0,24.0,9.0,6.0,13.0], dtype=dtype)

    solver.set_shape(n)
    solver.set_rcd_centralized(irn+1, jcn+1, a)
    solver._mumps_call(job=1)

    rhs = b.copy()
    solver._mumps_call(job=2)
    solver.set_rhs_centralized(rhs)
    solver._mumps_call(3)

    assert np.allclose(rhs, [1, 2, 3, 4, 5], atol=1e-10)
