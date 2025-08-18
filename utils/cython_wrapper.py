import re

def parse_c_struct(header_file, struct_name):
    """Extract struct fields from a C header file, including arrays and pointers."""
    
    with open(header_file, 'r') as f:
        lines = f.readlines()

    struct_pattern = rf"typedef struct\s*\{{(.*?)\}} {struct_name};"
    struct_match = re.search(struct_pattern, "".join(lines), re.DOTALL)

    if not struct_match:
        raise ValueError(f"Struct {struct_name} not found in {header_file}")

    struct_body = struct_match.group(1)

    fields = []
    for line in struct_body.split("\n"):
        line = line.strip()
        if not line or line.startswith("/*"):
            continue
        
        match = re.match(r"(\w+)\s+([\w,\s\*\[\]]+);", line)
        if match:
            ctype, variables = match.groups()
            variables = variables.split(",")

            for var in variables:
                var = var.strip()
                ptr = ""
                array_size = None

                if "*" in var:
                    ptr = "*"
                    var = var.replace("*", "").strip()

                array_match = re.match(r"(\w+)\[(\d+)\]", var)
                if array_match:
                    var, array_size = array_match.groups()
                
                fields.append((ctype, ptr, var, array_size))

    return fields


def generate_cython_wrapper(fields, class_name, filename):
    """Generates a full Cython .pyx file, handling scalars, pointers, and arrays."""

    cython_code = f"__all__ = ['{class_name}', '{filename}']\n\n"
    cython_code += "from libc.stdint cimport uintptr_t, int64_t\n"
    cython_code += "cdef extern from \"stddef.h\":\n"
    cython_code += "    ctypedef size_t size_t\n"
    cython_code += "    size_t sizeof(...)\n\n"

    cython_code += f"""cdef extern from '{filename}':  

    ctypedef int MUMPS_INT
    ctypedef double {class_name[0]}MUMPS_COMPLEX
    ctypedef double {class_name[0]}MUMPS_REAL
    ctypedef int64_t MUMPS_INT8

    char* MUMPS_VERSION

    ctypedef struct c_{class_name} '{class_name}':\n\n"""

    for ctype, ptr, varname, array_size in fields:
        if array_size:
            cython_code += f"        {ctype} {varname}[{array_size}]\n"
        else:
            cython_code += f"        {ctype}{ptr} {varname}\n"

    cython_code += f"\n    void c_{filename[:-2]} '{filename[:-2]}' (c_{class_name} *) nogil\n\n"

    cython_code += """def get_mumps_int_size():
    \"\"\"Return the size of MUMPS_INT in bytes.\"\"\"
    return sizeof(MUMPS_INT)\n\n"""

    cython_code += f"cdef class {class_name}:\n"
    cython_code += f"    cdef c_{class_name} obj\n\n"

    for ctype, ptr, varname, array_size in fields:
        if array_size:
            cython_code += f"""    property {varname}:
        def __get__(self):
            # Warning: direct memoryview on fixed-size array
            cdef {ctype}[:] view = self.obj.{varname}
            return view\n\n"""

        elif ptr:
            cython_code += f"""    property {varname}:
        def __get__(self): return <uintptr_t> self.obj.{varname}
        def __set__(self, value): self.obj.{varname} = <{ctype}*> (<uintptr_t> value)\n\n"""

        else:
            cython_code += f"""    property {varname}:
        def __get__(self): return self.obj.{varname}
        def __set__(self, value): self.obj.{varname} = value\n\n"""

    cython_code += f"""def {filename[:-2]}({class_name} s not None):
    with nogil:
        c_{filename[:-2]}(&s.obj)

__version__ = (<bytes> MUMPS_VERSION).decode('ascii')
"""
    return cython_code
