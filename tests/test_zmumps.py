import numpy as np
import pytest
import os
from mpi4py import MPI
from mumps4py.mumps_solver import MumpsSolver

# @pytest.mark.skipif("zmumps" not in os.getenv("MUMPS_SOLVERS", "").split(","), reason="zmumps not selected")
def test_solve_single():
    solver = MumpsSolver(verbose=False, system="complex128")

    dtype = np.complex128

    n = 5
    irn = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9], dtype=np.int32)
    jcn = np.array([0,2,4,0,3,5,1,3,4,0,7,8,1,6,9,2,6,7,3,8,9,4,7,9,5,6,8,6], dtype=np.int32)
    a = np.array([-1.+0.j, 1.+0.j,-1.+0.j, 1.+0.j, -1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j, 1.+0.j,  0.+10.j,
                  -1.+0.j,  1.+0.j,  0.+10.j, -1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j,  1.+0.j,  1.+0.j, -1.+0.j,
                  1.+0.j,  1.+0.j,  0.-10.j,  0.+10.j, 1.+0.j,  0.-10.j,  0.+10.j,  1.+0.j], dtype=dtype)
    b = np.array([[-10.,0.,0.,0.,0.,0.,0.,0.,0.,0.]], dtype=dtype)

    solver.set_rcd_centralized(irn+1, jcn+1, a, n)
    solver._mumps_call(job=1)

    rhs = b.copy()
    solver._mumps_call(job=2)
    solver.set_rhs_centralized(rhs)
    solver._mumps_call(3)

    if MPI.COMM_WORLD.Get_rank()==0:
        assert np.allclose(rhs, [0.04950495-4.95049505e-01j, -0.04950495+4.95049505e-01j, -5.-2.22044605e-16j, -4.9009901 -9.90099010e-01j, 4.95049505+4.95049505e-01j, -4.95049505-4.95049505e-01j, 0.+0.00000000e+00j, 5.+2.22044605e-16j, 0.04950495-4.95049505e-01j, 4.95049505+4.95049505e-01j], atol=1e-6)
