# MUMPS4PY
Python Interface for the MUMPS Solver

## Overview
MUMPS4PY is a Python wrapper for the **MUltifrontal Massively Parallel Solver (MUMPS)**, enabling efficient sparse matrix computations with parallel processing. It provides a high-performance interface for solving sparse linear systems using **MPI-based parallelism**.

<p align="center">
  <img src="docs/data/CvsP_2D.png" alt="HPC" width="48%">
  <img src="docs/data/CvsP_3D.png" alt="HPC" width="48%">
</p>

You can find more details, benchmarks, and examples in [Benchmarks](docs/Mumps4py_Benchmarks.pdf).

## Install via conda

You can install `mumps4py` on Linux, MacOS, and Windows using [conda/mamba](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):

```conda install mumps4py```


## Install manually
### Platform-Specific Installation Guides
- **Linux (Ubuntu)**: See instructions below.
- **Windows**: See [docs/install_windows.md](docs/install_windows.md).
- **macOS**: See [docs/install_macos.md](docs/install_macos.md).
- **Docker**: See [docs/docker.md](docs/docker.md).

### Dependencies
To use MUMPS4PY, you need:
- **MUMPS** (compiled with shared libraries)
- **MPI** (`mpich` or `openmpi`)
- **Python** 3.5+
- **NumPy**, **mpi4py**, and **SciPy**
Note on MUMPS Installation:
     - If you install MUMPS via apt (e.g., sudo apt install `libmumps-ptscotch-dev` as shown below), shared libraries are included, and you can follow this README directly.
     - If you download MUMPS from the official website (mumps-solver.org), you must build the shared libraries manually (e.g., `libdmumps.so`). See [docs/build_shared_libs.md](docs/build_shared_libs.md) for instructions on building shared libraries from the MUMPS source. Do not forgot to export all MUMPS library dependencies (eg. `libpord.so`, `libmetis.so`, etc).

#### Installing MUMPS (Ubuntu)
```bash
sudo apt install libmumps-ptscotch-dev
```

### Environment Variables

| Variable       | Description |
|----------------|------------------------------------------------------------------------|
| MUMPS_INC      | Path to the MUMPS include directory (e.g., ``dmumps_c.h``)             |
| MUMPS_LIB      |  Path to the MUMPS library directory (e.g., ``libdmumps.so``)          |
| MUMPS_SOLVERS  | Specify which MUMPS solver to use (``dmumps, cmumps, smumps, zmumps``) |

#### Example Setup:

```bash
export MUMPS_INC="/path/to/MUMPS/include"
export MUMPS_LIB="/path/to/MUMPS/lib"
export MUMPS_SOLVERS="dmumps"
```

##### You can configure mumps4py using all mumps solvers:

```bash
export MUMPS_SOLVERS="dmumps,cmumps,zmumps,smumps"
```

#### To make this permanent:

```bash
echo 'export MUMPS_INC="/path/to/MUMPS/include"' >> ~/.bashrc
echo 'export MUMPS_LIB="/path/to/MUMPS/lib"' >> ~/.bashrc
echo 'export MUMPS_SOLVERS="dmumps"' >> ~/.bashrc
source ~/.bashrc
```
### Installation

#### Clone and build:

```bash
git clone https://github.com/imadki/mumps4py.git
cd mumps4py
python setup.py build_ext --inplace
```
#### Install globaly

```bash
pip install .
```

#### Development mode
```bash
python3 -m pip install --user -e .
```

## Testing the Installation

### To verify that MUMPS4PY is installed correctly:

```bash
import mumps4py.mumps_solver as mps
solver = mps.MumpsSolver(verbose=True, system="double")
print("MUMPS Solver Initialized:", solver)
```

## Running Tests

### Run pytest to validate the installation:

```bash
pytest tests/
```
or
```bash
python -m pytest
```

### Example output:

```bash
tests/test_cmumps.py s  [25%]
tests/test_dmumps.py .  [50%]
tests/test_smumps.py s  [75%]
tests/test_zmumps.py s  [100%]
```

## Cleaning Build Artifacts

```bash
python setup.py clean
```

## Citation

```bash
@techreport{mumps4py2025,
  title = {Mumps4py: Python interface for the MUMPS solver},
  author = {I. Kissami},
  month = mar,
  year = {2025},
  institution = {Mohammed VI Polytechnic University},
  doi = {10.13140/RG.2.2.26885.05606/2}
}
```
