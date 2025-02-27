# error_handlings.py

MUMPS_ERROR_MESSAGES = {
    -1: "An error occurred on a processor. Check INFO(2) for details.",
    -2: "NNZ/NZ, NNZ loc/NZ loc, or P NNZ loc/P NZ loc are out of range.",
    -3: "Invalid JOB value. Ensure correct sequence: analysis (JOB=1) → factorization (JOB=2) → solve (JOB=3).",
    -4: "Error in user-provided permutation array PERM_IN at position INFO(2).",
    -5: "Real workspace allocation failed. Required size: INFO(2) real values.",
    -6: "Matrix is singular in structure. Structural rank: INFO(2).",
    -7: "Integer workspace allocation failed. Required size: INFO(2) integer values.",
    -8: "Internal integer workarray too small for factorization. Increase ICNTL(14).",
    -9: "Internal real/complex workarray too small. Increase ICNTL(14).",
    -10: "Numerically singular matrix or zero pivot encountered. INFO(2) holds eliminated pivots count.",
    -11: "Workarray too small for solution. Adjust LWK_USER.",
    -12: "Workarray too small for iterative refinement.",
    -13: "Workspace allocation failed during factorization/solve. Required size: INFO(2).",
    -14: "Internal integer workarray too small for solution. See error -8.",
    -15: "Integer workarray too small for iterative refinement/error analysis. See error -8.",
    -16: "N is out of range. INFO(2) = N.",
    -17: "Internal send buffer too small. Increase ICNTL(14).",
    -18: "ICNTL(27) is too large and may cause integer overflow. Max allowed: INFO(2).",
    -19: "ICNTL(23) is too small for factorization. Increase it.",
    -20: "Internal reception buffer too small. Increase ICNTL(14).",
    -21: "PAR=0 is invalid with only one processor. Set PAR=1 or increase the number of processors.",
    -22: "Invalid pointer array provided by the user. INFO(2) points to the incorrect array.",
    -23: "MPI was not initialized before calling MUMPS.",
    -24: "NELT is out of range. INFO(2) = NELT.",
    -25: "BLACS initialization failed. Try using Netlib BLACS version.",
    -26: "LRHS is out of range. INFO(2) = LRHS.",
    -27: "NZ_RHS and IRHS_PTR(NRHS+1) do not match.",
    -28: "IRHS_PTR(1) is not equal to 1. INFO(2) = IRHS_PTR(1).",
    -29: "LSOL_loc is too small. Should be ≥ INFO(23).",
    -30: "SCHUR_LLD is out of range. INFO(2) = SCHUR_LLD.",
    -31: "Invalid process grid for 2D block cyclic Schur complement. INFO(2) = MBLOCK - NBLOCK.",
    -32: "Incompatible values of NRHS and ICNTL(25). INFO(2) = NRHS.",
    -33: "ICNTL(26) requested in solve but Schur complement was not computed at analysis.",
    -34: "LREDRHS is out of range. INFO(2) = LREDRHS.",
    -35: "Schur expansion requested before reduction phase. Check ICNTL(26).",
    -36: "Incompatible values of ICNTL(25) and INFOG(28). INFO(2) = ICNTL(25).",
    -37: "ICNTL(25) is incompatible with another parameter. INFO(2) = ICNTL(xx).",
    -38: "Parallel analysis (ICNTL(28)=2) requires PT-SCOTCH or ParMetis.",
    -39: "Parallel analysis not possible with current matrix format and options.",
    -40: "Matrix marked positive definite (SYM=1) but a negative/null pivot was encountered. Use SYM=2.",
    -41: "Incompatible LWK_USER values between factorization and solution phases.",
    -42: "NRHS mismatch between analysis and solution phases. INFO(2) = NRHS.",
    -43: "Incompatible values of ICNTL(32) and ICNTL(xx).",
    -44: "Solve phase (JOB=3) cannot proceed due to missing factorization data. INFO(2) = ICNTL(31).",
    -45: "NRHS ≤ 0. INFO(2) contains NRHS value.",
    -46: "NZ_RHS ≤ 0 is not allowed with ICNTL(26)=1 and A⁻¹ requests.",
    -47: "Entries of A⁻¹ requested but NRHS ≠ N. INFO(2) = NRHS.",
    -48: "ICNTL(30) and ICNTL(xx) incompatible for A⁻¹.",
    -49: "SIZE_SCHUR is incorrect or was modified after analysis. INFO(2) = SIZE_SCHUR.",
    -50: "Error computing fill-reducing ordering. External ordering tool may have failed.",
    -51: "Graph size exceeds 2³¹-1 for 32-bit external ordering tools. INFO(2) = required size.",
    -52: "External ordering libraries require 64-bit integers but were compiled with 32-bit defaults.",
    -53: "Internal error due to inconsistent input data between consecutive calls.",
    -54: "ICNTL(35) changed between analysis and factorization. Re-run analysis with correct setting.",
    -55: "LRHS_loc too small or Nloc_RHS mismatch for distributed RHS.",
    -56: "RHS_loc and SOL_loc share memory but LRHS_loc < LSOL_loc.",
    -57: "Error in block matrix format. INFO(2) provides details.",
    -58: "ICNTL(48) issue: OpenMP not enabled or thread count mismatch.",
    -69: "Fortran INTEGER size mismatch with MUMPS_INT. Recompile with correct -DINTSIZE64 flag.",
    -70: "Save file already exists. Remove or change SAVE_PREFIX.",
    -71: "Error creating file for MUMPS save (JOB=7).",
    -72: "Error saving data (JOB=7); write operation failed (e.g., disk full).",
    -73: "Incompatible parameters between current and saved MUMPS instance.",
    -74: "File specified for restoring data (JOB=8) could not be opened.",
    -75: "Error restoring data (JOB=8); read operation failed.",
    -76: "Error deleting MUMPS files (JOB=-3); file not found or deletion failed.",
    -77: "Problem with SAVE_DIR or SAVE_PREFIX environment variables.",
    -78: "Workspace allocation issue during restore step.",
    -79: "No available Fortran file unit for MUMPS I/O.",
    -88: "SCOTCH ordering error. INFO(2) = SCOTCH error number.",
    -89: "SCOTCH k-way partitioning failed. Consider using METIS instead.",
    -90: "Error in out-of-core management. Check output on ICNTL(1)."
}

def get_mumps_error_message(error_code, info2=None):
    """Retrieve a formatted error message based on MUMPS error code."""
    message = MUMPS_ERROR_MESSAGES.get(error_code, "Unknown MUMPS error.")
    if info2 is not None:
        return f"MUMPS Error {error_code}: {message} INFO(2) = {info2}"
    return f"MUMPS Error {error_code}: {message}"
