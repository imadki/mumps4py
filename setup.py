#!/usr/bin/env python
# Author: Imad Kissami
# Contact: kissami.imad@gmail.com

"""
MUMPS for Python
"""

import sys
import os
import shutil
from setuptools import setup, Extension, Command
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.sdist import sdist
#from distutils import log  #Fix: Use distutils.log

try:
    import setuptools
except ImportError:
    setuptools = None

pyver = sys.version_info[:2]
if pyver < (2, 7) or (3, 0) <= pyver < (3, 5):
    raise RuntimeError("Python version 2.7 or >= 3.5 required")
if pyver == (2, 7):
    sys.stderr.write("WARNING: Python 2.7 is not supported.\n")


# --------------------------------------------------------------------
# ConfigCommand
# --------------------------------------------------------------------
class ConfigCommand(Command):
    """Custom config command (if needed)."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("Running configuration checks...")
        # You can add checks for dependencies, environment variables, etc.


# --------------------------------------------------------------------
# CleanCommand
# --------------------------------------------------------------------
class CleanCommand(Command):
    """Custom clean command to remove build artifacts."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        folders_to_remove = ['build', 'dist',
                             'mumps4py.egg-info']
        for folder in folders_to_remove:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"Removed {folder}")

        for root, dirs, files in os.walk("mumps4py"):
            for file in files:
                if file.endswith((".so", ".pyd", ".c", ".o", ".pyx")):
                    os.remove(os.path.join(root, file))
                    print(f"Removed {file}")


# --------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------

topdir = os.path.abspath(os.path.dirname(__file__))

metadata = {
    "name": "mumps4py",
    "version": "0.1.1rc0",
    "description": "MUMPS for Python",
    "long_description": open(os.path.join(topdir, 'README.md')).read() if \
    os.path.exists(os.path.join(topdir, 'README.md')) else "A Python interface to the MUMPS solver library",
    "long_description_content_type":"text/markdown",
    "keywords": ['MUMPS', 'solver', 'MPI'],
    "url": "https://github.com/imadki/mumps4py",
    "download_url": "https://pypi.io/packages/source/m/mumps4py/mumps4py-0.1.0.tar.gz",
    "provides": ['mumps4py'],
    "requires": ['numpy', 'mpi4py'],
    "classifiers": [
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Mathematics'],

}

CYTHON_VERSION = '0.29.26'

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------

def check_cython(version):
    """Check if Cython is installed and meets the minimum version requirement."""
    from distutils.version import StrictVersion
    try:
        import Cython
        cython_ver = getattr(Cython, '__version__', None) or getattr(Cython.Compiler.Version, 'version', None)
        if StrictVersion(cython_ver) < StrictVersion(version):
            sys.stderr.write(f"Error: Cython {cython_ver} found, but {version} is required.\n")
            return False
        return True
    except ImportError:
        sys.stderr.write(f"Error: Cython not found. Please install Cython>={version}.\n")
        return False

def run_cython(source, includes=(), force=False):
    """Run Cython on a .pyx file to generate a .c file."""
    from Cython.Build import cythonize
    from distutils.errors import DistutilsError
    target = os.path.splitext(source)[0] + '.c'

    if not check_cython(CYTHON_VERSION):
        raise DistutilsError(f"Cython>={CYTHON_VERSION} is required.")

    #log.info(f"Cythonizing '{source}' -> '{target}'")
    cythonize([source], include_path=includes, force=force)

# --------------------------------------------------------------------
# Extension modules
# --------------------------------------------------------------------

def get_ext_modules():
    import platform

    wrapper_script = os.path.join(topdir, 'utils', 'cython_wrapper.py')
    if not os.path.exists(wrapper_script):
        raise FileNotFoundError(f"Wrapper script not found: {wrapper_script}")

    namespace = {}
    with open(wrapper_script, 'r') as f:
        exec(f.read(), namespace)
    parse_c_struct = namespace['parse_c_struct']
    generate_cython_wrapper = namespace['generate_cython_wrapper']

    # Détection de la plateforme
    SYSTEM = platform.system().lower()  # 'windows', 'linux', 'darwin' pour macOS

    # Use environment variables for external paths
    MUMPS_INCLUDE_DIR = os.environ.get('MUMPS_INC', "")
    MUMPS_LIB_DIR = os.environ.get('MUMPS_LIB', "")

    if not MUMPS_INCLUDE_DIR or not MUMPS_LIB_DIR:
        raise RuntimeError("MUMPS_INC and MUMPS_LIB must be set before using mumps4py.")


    mumps_solvers = os.environ.get('MUMPS_SOLVERS', 'dmumps').split(',')
    mumps_libraries = mumps_solvers = [s.strip() for s in mumps_solvers if s.strip()] or ['dmumps']

    # # Map solver names to MSYS2-style names if needed
    # solver_map = {
    #     'cmumps': 'mumps-cso',  # Complex single precision, sequential
    #     'dmumps': 'mumps-dso',  # Double precision, sequential
    #     'smumps': 'mumps-sso',  # Single precision, sequential
    #     'zmumps': 'mumps-zso',  # Complex double precision, sequential
    # }

    # # Adjust solver names for MSYS2 (e.g., dmumps -> dso)
    # if SYSTEM == 'windows':
    #     mumps_libraries = [solver_map.get(solver, solver) for solver in mumps_solvers]

    # Check for header files
    for solver in mumps_solvers:
        header_file = os.path.join(MUMPS_INCLUDE_DIR, f'{solver}_c.h')
        if not os.path.exists(header_file):
            sys.stderr.write(f"Header file not found: {header_file}\n")
            sys.stderr.write("Please ensure MUMPS is installed and set MUMPS_INC to the correct path.\n")
            sys.exit(1)

    # Check for solver libraries
    for solver in mumps_libraries:
        lib_found = False
        for ext in ['.a', '.dll.a','.lib'] if SYSTEM == 'windows' else ['.so', '.a','.dylib']:
            lib_file = os.path.join(MUMPS_LIB_DIR, f'lib{solver}{ext}' if SYSTEM != 'windows' else f'{solver}{ext}')
            if os.path.exists(lib_file):
                lib_found = True
                break
        if not lib_found:
            sys.stderr.write(f"Library not found for '{solver}' in {MUMPS_LIB_DIR}\n")
            sys.stderr.write(f"Please ensure MUMPS solver libraries (e.g., lib{solver}_ptscotch-5.6.1{ext}) are installed and set MUMPS_LIB correctly.\n")
            sys.exit(1)

    # Generate wrapper
    mumps4py_dir = 'mumps4py'
    os.makedirs(mumps4py_dir, exist_ok=True)
    output_file = os.path.join(mumps4py_dir, '_mumps_wrapper.pyx')

    cython_code = ''
    for solver in mumps_solvers:
        header_file = os.path.join(MUMPS_INCLUDE_DIR, f'{solver}_c.h')
        struct_name = f'{solver.upper()}_STRUC_C'
        try:
            fields = parse_c_struct(header_file, struct_name)
            wrapper_code = generate_cython_wrapper(fields, class_name=struct_name, filename=f'{solver}_c.h')
            cython_code += wrapper_code + '\n\n'
        except Exception as e:
            sys.stderr.write(f"Error generating wrapper for {solver}: {e}\n")
            sys.exit(1)

    with open(output_file, 'w') as f:
        f.write(cython_code)

    # Define the extension with external paths
    if SYSTEM == 'windows':
        extra_link_args = []
    elif SYSTEM == 'darwin':
        extra_link_args = ['-Wl']
    elif  SYSTEM == 'linux':
        extra_link_args = ['-Wl,--allow-multiple-definition']

    return [Extension('mumps4py._mumps_wrapper',
                      sources=[output_file],
                      include_dirs=[MUMPS_INCLUDE_DIR],
                      library_dirs=[MUMPS_LIB_DIR],
                      libraries=mumps_libraries + ['mumps_common'],
                      runtime_library_dirs=[MUMPS_LIB_DIR] if SYSTEM != "windows" else None,
                      extra_link_args=extra_link_args if SYSTEM != 'windows' else None)]

# --------------------------------------------------------------------
# Custom Commands
# --------------------------------------------------------------------
class BuildSources(build):
    """Build sources, ensuring Cython compilation if necessary."""
    def run(self):
        run_cython(os.path.join('mumps4py', '_mumps_wrapper.pyx'), includes=[os.getenv('MUMPS_INC', '/usr/include')], force=self.force)
        build.run(self)

def run_setup():
    setup(
        packages=['mumps4py'],
        package_dir={'mumps4py': 'mumps4py'},
        package_data={'mumps4py': ['*.h', '*.pxd', '*.pyx']},
        ext_modules=get_ext_modules(),
        cmdclass={
            'config': ConfigCommand,
            'build': build,
            'build_src': BuildSources,
            'build_ext': build_ext,
            'install': install,
            'clean': CleanCommand,
            'sdist': sdist,
        },
        install_requires=['numpy', 'mpi4py', f'Cython>={CYTHON_VERSION}'],
        zip_safe=False,
        **metadata
    )

def build_sources(cmd):
    MUMPS_INCLUDE_DIR = os.environ.get('MUMPS_INCLUDE', '/usr/include')
    source = os.path.join('mumps4py', '_mumps_wrapper.pyx')
    depends = []
    includes = [MUMPS_INCLUDE_DIR]
    run_cython(source, depends, includes, wdir='mumps4py', force=cmd.force, VERSION=CYTHON_VERSION)

BuildSources.run = build_sources


# --------------------------------------------------------------------
def main():
    run_setup()

if __name__ == '__main__':
    main()
