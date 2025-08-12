import ctypes
import numpy as np
from mpi4py import MPI

from mumps4py.error_handlings import get_mumps_error_message

class mumps_double_complex(ctypes.Structure):
    _fields_ = [("r", ctypes.c_double), ("i", ctypes.c_double)]

class mumps_complex(ctypes.Structure):
    _fields_ = [("r", ctypes.c_float), ("i", ctypes.c_float)]

class MumpsSolver:
    
    def __init__(self, system='double', sym=0, par=1, verbose=None, 
                 mpi_comm=None, mem_relax=20):
        
        """
        Initialize the MUMPS solver.

        :param sym: Symmetric (1) or Unsymmetric (0) matrix.
        :param par: Parallel mode (1 = master process, 0 = slave process).
        :param system: Precision type ("double", "single", "complex128", "complex64")
        """
        
        self.system = system
        
        self._precompute_mumps_func()
        self._load_mumps_module()
        self._transform_array_to_pointer()
        self.int_size = self._detect_mumps_int_type()
        
        # Set MPI communication
        self.comm = MPI.COMM_WORLD if mpi_comm is None else mpi_comm
        self.struct.comm_fortran = self.comm.py2f()
        self.struct.sym = sym
        self.struct.par = par
        self.struct.icntl[13] = mem_relax  # Memory relaxation

        # Initialize MUMPS
        self._mumps_c.restype = None
        self._init_mumps()
        self.set_silent()

        # Enable verbose mode if requested
        if verbose:
            self.set_verbose()
            
    def _precompute_mumps_func(self):
        
        """Precompute the MUMPS function name based on system type."""
        self.mumps_variants = {
            "double": ("dmumps_c", "DMUMPS_STRUC_C"),
            "single": ("smumps_c", "SMUMPS_STRUC_C"),
            "complex128": ("zmumps_c", "ZMUMPS_STRUC_C"),
            "complex64": ("cmumps_c", "CMUMPS_STRUC_C")
        }
    
        if self.system not in self.mumps_variants:
            raise ValueError(
            f"Unsupported system type: {self.system}. "
            f"Supported types are: {list(self.mumps_variants.keys())}"
        )
        
        self.mumps_func, self.struct_type =self.mumps_variants[self.system]
        
    def _transform_array_to_pointer(self):
        """Configure the MUMPS system based on precision and type."""
        
        self.transform_arr = lambda arr: (
            ctypes.cast(arr.ctypes.data, ctypes.c_void_p).value
            if not (np.issubdtype(arr.dtype, np.integer) and arr.dtype != self.int_size)
            else exec("raise TypeError(f'Array types differ: MUMPS expect {self.int_size}, but got {arr.dtype}')")
        )

    def _load_mumps_module(self):
        """Dynamically load the MUMPS module and stop execution if missing."""
        try:
            # Try importing the MUMPS wrapper module
            module = __import__("mumps4py._mumps_wrapper", fromlist=[self.mumps_func, self.struct_type])
    
            # Ensure the required MUMPS function exists
            if not hasattr(module, self.mumps_func):
                raise ImportError(f"Error: MUMPS function '{self.mumps_func}' is missing in mumps_wrapper.")
    
            # Ensure the required MUMPS structure exists
            if not hasattr(module, self.struct_type):
                raise ImportError(f"Error: MUMPS structure '{self.struct_type}' is missing in mumps_wrapper.")
    
            # Assign MUMPS function and structure
            self._mumps_c = getattr(module, self.mumps_func)
            self.struct = getattr(module, self.struct_type)()
    
        except ImportError as e:
            raise RuntimeError(f"MUMPS module loading failed: {e}")  # 🚀 Stops execution immediately!
  
    
    def _detect_mumps_int_type(self):
        """Detect the integer type expected by MUMPS based on MUMPS_INT size."""
        from mumps4py._mumps_wrapper import get_mumps_int_size
        int_size = get_mumps_int_size()

        if int_size == 4:
            return np.int32
        elif int_size == 8:
            return np.int64
        else:
            raise RuntimeError(f"Unexpected MUMPS_INT size: {int_size} bytes")
        
    def _init_mumps(self):
        """Initialize the MUMPS solver."""
        self.struct.job = -1  # Initialize MUMPS
        self._mumps_c(self.struct)#(ctypes.byref(self.struct))
        if self.struct.infog[0] < 0:
            raise RuntimeError(f"MUMPS initialization failed: {get_mumps_error_message(self.struct.infog[0], self.struct.infog[1])}")
    
    def set_silent(self):
        self.struct.icntl[0] = -1  # suppress error messages
        self.struct.icntl[1] = -1  # suppress diagnostic messages
        self.struct.icntl[2] = -1  # suppress global info
        self.struct.icntl[3] = 0

    def set_verbose(self):
        self.struct.icntl[0] = 6  # error messages
        self.struct.icntl[1] = 0  # diagnostic messages
        self.struct.icntl[2] = 6  # global info
        self.struct.icntl[3] = 2
        
    def set_icntl(self, idx, val):
        """Set the icntl value.
        The index should be provided as a 1-based number.
        """
        self.struct.icntl[idx-1] = val
        
    def set_cntl(self, idx, val):
        """Set the icntl value.
        The index should be provided as a 1-based number.
        """
        self.struct.cntl[idx-1] = val

    def set_coo_centralized(self, mtx):
        """
        Set the sparse matrix.

        Parameters
        ----------
        mtx : scipy sparse martix
            The sparse matrix in COO format.
        """
        assert mtx.shape[0] == mtx.shape[1]

        rr = mtx.row + 1
        cc = mtx.col + 1
        data = mtx.data

        if self.struct.sym > 0:
            idxs = np.where(cc >= rr)[0]  # upper triangular matrix
            rr, cc, data = rr[idxs], cc[idxs], data[idxs]

        self.set_rcd_centralized(rr, cc, data, mtx.shape[0])
    
    ####################################################################
    # Centralized (master process enters the all matrix)
    ####################################################################
    def set_rcd_centralized(self, ir, ic, data, n):
        """
        Set the matrix by row and column indicies and data vector.
        The matrix shape is determined by the maximal values of
        row and column indicies. The indices start with 1.

        Parameters
        ----------
        ir : array
            The row idicies.
        ic : array
            The column idicies.
        data : array
            The matrix entries.
        n : int
            The matrix dimension.
        """
        assert ir.shape[0] == ic.shape[0] == data.shape[0]
        self.struct.n = n  # Matrix size
        self.struct.nz = ir.shape[0]
        if hasattr(self.struct, 'nnz'):
            self.struct.nnz = ir.shape[0]
            
        self.struct.irn = self.transform_arr(ir)
        self.struct.jcn = self.transform_arr(ic)
        self.struct.a   = self.transform_arr(data)
        
    ####################################################################
    # Centralized data (master process enters the all matrix)
    ####################################################################
    def set_data_centralized(self, data, n):
        """
        Set the matrix data vector.
        The matrix shape is determined by the maximal values of
        row and column indicies. The indices start with 1.

        Parameters
        ----------
        data : array
            The matrix entries.
        n : int
            The matrix dimension.
        """
        self.struct.n = n  # Matrix size
        self.struct.nz = data.shape[0]
        if hasattr(self.struct, 'nnz'):
            self.struct.nnz = data.shape[0]
            
        self.struct.a   = self.transform_arr(data)
        
    ####################################################################
    # Distributed (each process enters some portion of the matrix)
    ####################################################################
    def set_rc_distributed(self, ir_loc, ic_loc, n):
        """Set the distributed assembled row col.
    
        Distributed assembled matrices require setting icntl(18) != 0.
        """
        
        """
        Set the matrix by row and column indicies and data vector.
        The matrix shape is determined by the maximal values of
        row and column indicies. The indices start with 1.
    
        Parameters
        ----------
        ir_loc : local array
            The row idicies.
        ic_loc : local array
            The column idicies.
        n : int
            The matrix dimension. (set by the rank 0)
        """
        assert ir_loc.shape[0] == ic_loc.shape[0] 
        if self.comm.Get_rank() == 0:
            self.struct.n = n  # Matrix size
        self.struct.nz_loc = ir_loc.shape[0]
        
        if hasattr(self.struct, 'nnz_loc'):
            self.struct.nnz_loc = ir_loc.shape[0]
        
        self.struct.irn_loc = self.transform_arr(ir_loc)
        self.struct.jcn_loc = self.transform_arr(ic_loc)
        
    ####################################################################
    # Distributed (each process enters some portion of the matrix)
    ####################################################################
    def set_rcd_distributed(self, ir_loc, ic_loc, data_loc, n):
        """Set the distributed assembled matrix.

        Distributed assembled matrices require setting icntl(18) != 0.
        """
        
        """
        Set the matrix by row and column indicies and data vector.
        The matrix shape is determined by the maximal values of
        row and column indicies. The indices start with 1.

        Parameters
        ----------
        ir_loc : local array
            The row idicies.
        ic_loc : local array
            The column idicies.
        data_loc : local array
            The matrix entries.
        n : int
            The matrix dimension. (set by the rank 0)
        """
        assert ir_loc.shape[0] == ic_loc.shape[0] == data_loc.shape[0]
        if self.comm.Get_rank() == 0:
            self.struct.n = n  # Matrix size
        self.struct.nz_loc = ir_loc.shape[0]
        
        if hasattr(self.struct, 'nnz_loc'):
            self.struct.nnz_loc = ir_loc.shape[0]
        
        self.struct.irn_loc = self.transform_arr(ir_loc)
        self.struct.jcn_loc = self.transform_arr(ic_loc)
        self.struct.a_loc   = self.transform_arr(data_loc)
    
    ####################################################################
    # Distributed data (each process enters some portion of the matrix)
    ####################################################################
    def set_data_distributed(self, data_loc, n):
        """Set the distributed assembled matrix.

        Distributed assembled matrices require setting icntl(18) != 0.
        """
        
        """
        Set the matrix data vector.
        The matrix shape is determined by the maximal values of
        row and column indicies. The indices start with 1.

        Parameters
        ----------
        data_loc : local array
            The matrix entries.
        n : int
            The matrix dimension.
        """
        if self.comm.Get_rank() == 0:
            self.struct.n = n  # Matrix size
        self.struct.nz_loc = data_loc.shape[0]
        
        if hasattr(self.struct, 'nnz_loc'):
            self.struct.nnz_loc = data_loc.shape[0]
        
        self.struct.a_loc   = self.transform_arr(data_loc)
        
    ####################################################################
    # Elemental centralized matrix (master process)
    ####################################################################
    def set_elemental_matrix(self, eltptr, eltvar, a_elt, n, nelt):
        """
        Set the matrix in elemental format (ICNTL(5)=1 and ICNTL(18)=0).
        
        Parameters
        ----------
        n : int
            The order of the matrix (N).
        nelt : int
            The number of elements (NELT).
        eltptr : array (int)
            Pointer array to the first variable of each element (ELTPTR).
        eltvar : array (int)
            List of element variables (ELTVAR).
        a_elt : array (float or complex)
            The values of elements stored by column (A_ELT).
        """
        assert eltptr.shape[0] == nelt + 1, "ELTPTR must have size NELT + 1"
        assert eltvar.shape[0] == eltptr[-1] - 1, "ELTVAR size mismatch"
    
        # Ensure MUMPS is in elemental format mode
        if self.struct.icntl[4] != 1:  # ICNTL(5) should be at index 4 (zero-based)
            raise ValueError("ICNTL(5) must be set to 1 for elemental format.")
        if self.struct.icntl[17] != 0:  # ICNTL(18) should be at index 17
            raise ValueError("ICNTL(18) must be set to 0 for centralized input.")
    
        # Store matrix properties
        self.struct.n = n
        self.struct.nelt = nelt
    
        # Convert arrays to proper ctypes pointers
        self.struct.eltptr = self.transform_arr(eltptr)
        self.struct.eltvar = self.transform_arr(eltvar)
        self.struct.a_elt  = self.transform_arr(a_elt)

    ####################################################################
    # Centralized rhs (master process)
    ####################################################################
    def set_rhs_centralized(self, rhs):
        
        #assert rhs.shape[1] == self.struct.n, "rhs should have size of n" 
        """Set the right hand side of the linear system."""
        if len(rhs.shape) > 1:
            assert rhs.shape[1] == self.struct.n, "RHS column size must match matrix dimension"
            self.struct.nrhs = rhs.shape[0]
            self.struct.lrhs = rhs.shape[1]
        else:
            assert rhs.shape[0] == self.struct.n, "RHS size must match matrix dimension"
            self.struct.nrhs = 1
        
        self.struct.rhs = self.transform_arr(rhs)
    
    ####################################################################
    # Centralized rhs (master process)
    ####################################################################
    def set_rhs_distributed(self, rhs_loc, irhs_loc, nrhs):
        """
        Set up the distributed right-hand side (RHS) for MUMPS.
    
        Parameters
        ----------
        rhs_loc : np.array
            The local right-hand side values.
        irhs_loc : np.array
            The mapping of local rows to global indices (1-based for MUMPS).
        nrhs : int
            Number of right-hand side vectors.
        """
        assert rhs_loc.shape[0] == irhs_loc.shape[0], "Mismatch in local RHS size and index mapping"
    
        if self.struct.icntl[19] not in [10, 11]:  
            raise ValueError("ICNTL(20) must be set to 10 or 11 for distributed rhs.")
    
        # ✅ Set distributed RHS parameters in MUMPS structure
        self.struct.nrhs = nrhs  # Number of right-hand side vectors
        self.struct.nloc_rhs = rhs_loc.shape[0]  # Local number of RHS rows
        self.struct.lrhs_loc = rhs_loc.shape[0]  # Leading dimension (must be ≥ nloc_rhs)
    
        # Convert arrays to proper ctypes pointers
        self.struct.irhs_loc = self.transform_arr(irhs_loc)
        self.struct.rhs_loc  = self.transform_arr(rhs_loc)
        
    ###################################################################
    # Centralized rhs (master process)
    ####################################################################
    def set_rhs_sparse(self, rhs_sparse, rhs_row_indices, rhs_col_ptr, nrhs):
        """
        Set the sparse right-hand side (RHS) for MUMPS.
    
        Parameters
        ----------
        rhs_values : np.array
            The numerical values of non-zero entries in RHS.
        rhs_row_indices : np.array
            The row indices of the non-zero entries.
        rhs_col_ptr : np.array
            Column pointers indicating where each RHS column starts.
        nrhs : int
            Number of right-hand side vectors.
        """
        assert rhs_sparse.shape[0] == rhs_row_indices.shape[0], "Mismatch in RHS non-zero values and row indices"
        assert rhs_col_ptr.shape[0] == nrhs + 1, "Column pointer array must have size NRHS+1"
        
        if self.struct.icntl[19] not in [1, 2, 3]:  
            raise ValueError("ICNTL(20) must be set to 1, 2 or 3 for sparse rhs.")
    
        # ✅ Set sparse RHS parameters in MUMPS structure
        self.struct.nz_rhs = rhs_sparse.shape[0]  # Total non-zeros in RHS
        self.struct.nrhs = nrhs  # Number of RHS vectors
    
        # Convert arrays to proper ctypes pointers
        self.struct.rhs_sparse  = self.transform_arr(rhs_sparse)
        self.struct.irhs_sparse = self.transform_arr(rhs_row_indices)
        self.struct.irhs_ptr    = self.transform_arr(rhs_col_ptr)
        

    def enable_distributed_solution(self, nrhs):
        """
        Configure MUMPS to store the solution in a distributed format.
    
        Parameters
        ----------
        nrhs : int
            Number of right-hand side vectors.
        """
        
        # Enable distributed solution storage (ICNTL(21) = 1)
        if self.struct.icntl[20] !=1 :  
            raise ValueError("ICNTL(21) must be set to 1 for distributed sol.")
    
        # ✅ Retrieve INFO(23) after factorization to determine local solution size
        info_23 = self.struct.info[22]  # INFO(23) contains number of local solution variables
        
        if info_23 <= 0:
            raise RuntimeError("Invalid INFO(23) value. Ensure factorization (JOB=2) was done.")
    
        # Allocate ISOL_loc based on INFO(23) (determines local solution mapping)
        isol_loc = np.zeros(info_23, dtype=np.int32)  # Placeholder
        self.struct.isol_loc = self.transform_arr(isol_loc)
    
        # Allocate SOL_loc for storing the distributed solution
        sol_loc = np.zeros((info_23, nrhs), dtype=np.float64)
        self.struct.lsol_loc = info_23  # Leading dimension for solution
        self.struct.sol_loc  = self.transform_arr(sol_loc)

    def write_matrix_to_file(self, filename):
        """
        Configure MUMPS to write the matrix and RHS to a file during the analysis phase.
        """
        if len(filename) > 1023:
            raise ValueError("Filename is too long. Maximum allowed length is 1023 characters.")
        self.struct.write_problem = filename.encode('utf-8')
        
    def get_schur(self, schur_list):
        """Get the Schur matrix and the condensed right-hand side vector.

        Parameters
        ----------
        schur_list : array
            The list of the Schur DOFs (indexing starts with 1).

        Returns
        -------
        schur_arr : array
            The Schur matrix of order 'schur_size'.
        schur_rhs : array
            The reduced right-hand side vector.
        """
        # Schur
        slist = schur_list + 1
        schur_size = slist.shape[0]
        schur_arr = np.empty((schur_size**2, ), dtype='d')
        schur_rhs = np.empty((schur_size, ), dtype='d')
        self._schur_rhs = schur_rhs

        self.struct.size_schur = schur_size
        self.struct.listvar_schur = self.transform_arr(slist)
        self.struct.schur         = self.transform_arr(schur_arr)
        
        self.struct.lredrhs = schur_size
        self.struct.redrhs = self.transform_arr(schur_rhs)

        # get matrix
        self.struct.schur_lld = schur_size
        self.struct.nprow = 1
        self.struct.npcol = 1
        self.struct.mblock = 100
        self.struct.nblock = 100

        self.struct.icntl[18] = 3  # centr. Schur complement stored by columns
        self.struct.job = 4  # analyze + factorize
        self._mumps_c(ctypes.byref(self.struct))

        # get RHS
        self.struct.icntl[25] = 1  # Reduction/condensation phase
        self.struct.job = 3  # solve
        self._mumps_c(ctypes.byref(self.struct))

        return schur_arr.reshape((schur_size, schur_size)), schur_rhs

    def expand_schur(self, x2):
        """Expand the Schur local solution on the complete solution.

        Parameters
        ----------
        x2 : array
            The local Schur solution.

        Returns
        -------
        x : array
            The global solution.
        """
        self._schur_rhs[:] = x2
        self.struct.icntl[25] = 2  # Expansion phase
        self.struct.job = 3  # solve
        self._mumps_c(ctypes.byref(self.struct))

        return self.pointer_to_numpy(self.struct.rhs, self.dtype, (self.struct.n,))


    def analyze(self):
        """Analyze the Matrix before factorization."""
        self._mumps_call(job=1)
    
    def factorize(self):
        """Perform LU or Cholesky factorization."""
        self._mumps_call(job=2)
        
    def solve(self):
        """Solve Ax=b."""
        self._mumps_call(job=3)
    
    def __call__(self, job):
        """Set the job and call MUMPS."""
        self._mumps_call(job)
        
    def _mumps_call(self, job):
        """Set the job and call MUMPS with error handling."""
        self.struct.job = job
        self._mumps_c(self.struct)

        # Vérification des erreurs après l'appel à MUMPS
        error_code = self.struct.infog[0]
        if error_code < 0:
            info2 = self.struct.infog[1]  # INFO(2) peut être utile pour certains messages
            raise RuntimeError(get_mumps_error_message(error_code, info2))
    
    def _check_error(self):
        """Check MUMPS error codes."""
        if self.struct.info[0] < 0:
            raise RuntimeError(f"MUMPS Error {self.struct.info[0]}")
            
    def __del__(self):
        """Finish MUMPS cleanup safely."""
        if hasattr(self, 'struct') and self.struct is not None:
            self._mumps_call(job=-2)  # Terminate MUMPS

        self.struct = None
         
    def __repr__(self):
        return f"<MumpsSolver system={self.system}, sym={self.struct.sym}, n={self.struct.n}>"
    
    
    @staticmethod 
    def _raise_type_error(found, expected):
        raise TypeError(f"vector types differ: expected {expected}, but got {found}")

    @staticmethod 
    def pointer_to_numpy(ptr, dtype, shape):
        """
        Convert a ctypes pointer (integer address) to a NumPy array.

        :param ptr: Integer address of the pointer.
        :param dtype: NumPy dtype (e.g., np.float64, np.int32).
        :param shape: Tuple indicating the shape of the array.
        :return: NumPy array view of the data.
        """
        if not ptr:
            raise ValueError("Pointer is NULL. Cannot convert to NumPy array.")

        # Map NumPy dtypes to ctypes
        ctype_map = {
            np.float64: ctypes.c_double,
            np.float32: ctypes.c_float,
            np.int32: ctypes.c_int32,
            np.int64: ctypes.c_int64
        }

        if dtype not in ctype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        ctype = ctype_map[dtype]

        # Create a ctypes array type
        array_type = ctypes.POINTER(ctype * np.prod(shape))
        ctypes_array = ctypes.cast(ptr, array_type).contents

        # Convert to NumPy array
        return np.ctypeslib.as_array(ctypes_array).reshape(shape)

