# Contributing to mumps4py

Thank you for your interest in contributing to **mumps4py**! Contributions of all
kinds are welcome: bug reports, feature requests, documentation improvements and
code.

## Reporting bugs and requesting features

Please use the [GitHub issue tracker](https://github.com/imadki/mumps4py/issues).
When reporting a bug, include:

- your operating system and Python version,
- the MUMPS version and how it was installed (`apt`, source build, conda, …),
- the values of the `MUMPS_INC`, `MUMPS_LIB` and `MUMPS_SOLVERS` environment
  variables,
- a minimal example that reproduces the problem, and
- the full error message or traceback.

## Setting up a development environment

```bash
git clone https://github.com/imadki/mumps4py.git
cd mumps4py
# Make sure MUMPS shared libraries and MPI are available (see README.md).
python -m pip install --user -e .
```

Build the Cython extension in place:

```bash
python setup.py build_ext --inplace
```

## Running the tests

```bash
pytest tests/
```

The test suite exercises all four MUMPS drivers (`smumps`, `dmumps`, `cmumps`,
`zmumps`). Tests for a driver are skipped automatically when the corresponding
MUMPS library is not available in your `MUMPS_SOLVERS` configuration.

## Submitting changes

1. Fork the repository and create a topic branch from `main`.
2. Make your change, keeping the coding style consistent with the surrounding
   code.
3. Add or update tests when you fix a bug or add a feature.
4. Ensure `pytest tests/` passes locally.
5. Open a pull request describing the motivation and the change.

## Code of conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).
