"""
Backend for generating the run-time program.
"""

# TODO: Consider using namespaces?

import numpy as np
import ctypes
import os.path
import subprocess
import tempfile
from .inputs import LinearInput, LogarithmicInput

class Codegen1D:
    def __init__(self, name, table, conserve_sum, float_dtype, target):
        self.name           = str(name)
        self.table          = table.table
        self.input1         = table.input1
        self.state_names    = table.state_names
        self.num_states     = int(table.num_states)
        self.order          = int(table.order)
        self.conserve_sum   = conserve_sum
        self.initial_state  = table.exact.initial_state(conserve_sum)
        self.target         = str(target)
        self.float_dtype    = float_dtype
        self._cached_load   = None
        assert self.num_states >= 0
        assert self.order >= 0
        assert self.target in ('host', 'cuda')
        assert self.float_dtype in (np.float32, np.float64)
        self.source_code = (
                self._preamble() +
                self._initial_state() +
                self._table_data() +
                self._kernel() +
                self._entrypoint())

    def _preamble(self):
        c = ""
        if self.target == 'host':
            c += "#include <math.h>       /* log2 */\n\n"
        if self.float_dtype == np.float32:
            c +=  "typedef float real;\n\n"
        elif self.float_dtype == np.float64:
            c +=  "typedef double real;\n\n"
        else: raise NotImplementedError(self.float_dtype)
        return c

    def _initial_state(self):
        c = ""
        for name, value in zip(self.state_names, self.initial_state):
            c += f"const real INITIAL_{name} = {value};\n"
        c += "\n"
        return c

    def _table_data(self):
        table_size = self.input1.num_buckets * self.num_states**2 * (self.order+1)
        table_data = self.table.transpose(0, 2, 1, 3)
        assert table_data.size == table_size
        table_data = np.array(table_data, dtype=self.float_dtype)
        data_str   = ',\n    '.join((','.join(str(x) for x in bucket.flat)) for bucket in table_data)
        if self.target == 'cuda':
            device = '__device__ '
        elif self.target == 'host':
            device = ''
        return f"{device}const real table[{table_size}] = {{\n    {data_str}}};\n\n"

    def _entrypoint(self):
        c = 'extern "C" '
        if self.target == 'cuda':
            c += "__global__ "
        c += f"void {self.name}_advance(int n_inst, "
        c += f"real* {self.input1.name}, int* {self.input1.name}_indices, "
        c +=  ", ".join(f"real* {state}" for state in self.state_names)
        c +=  ") {\n"
        if self.target == 'host':
            c +=  "    for(int index = 0; index < n_inst; ++index) {\n"
        elif self.target == 'cuda':
            c +=  "    const int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
            c +=  "    if( index >= n_inst ) { return; }\n"
        c +=  "        // Access this instance's data.\n"
        access_input = f"{self.input1.name}[{self.input1.name}_indices[index]]"
        c += f"        real* state[{self.num_states}] = {{{', '.join(self.state_names)}}};\n"
        c += f"        for(int x = 0; x < {self.num_states}; ++x) {{ state[x] += index; }}\n"
        c += f"        {self.name}_kernel({access_input}, state);\n"
        if self.target == 'host':
            c +=  "    }\n"
        c +=  "}\n\n"
        return c

    def _kernel(self):
        c = ""
        if self.target == 'cuda':
            c += "__device__ "
        c += f"__inline__ void {self.name}_kernel("
        c += f"real input, real* state[{self.num_states}]) {{\n"
        c +=  "    const real* tbl_ptr = table;\n"
        c +=  "    // Locate the input within the look-up table.\n"
        if isinstance(self.input1, LinearInput):
            c += f"    input = (input - {self.input1.minimum}) * {self.input1.bucket_frq};\n"
        elif isinstance(self.input1, LogarithmicInput):
            if self.target == 'host':
                c += f"    input = log2(input + {self.input1.scale});\n"
            elif self.target == 'cuda':
                if self.float_dtype == np.float32:
                    c += f"    input = log2f(input + {self.input1.scale});\n"
                elif self.float_dtype == np.float64:
                    c += f"    input = log2(input + {self.input1.scale});\n"
                else: raise NotImplementedError(self.float_dtype)
            else: raise NotImplementedError(self.target)
            c += f"    input = (input - {self.input1.log2_minimum}) * {self.input1.bucket_frq};\n"
        else: raise NotImplementedError(type(self.input1))
        c +=  "    int bucket = (int) input;\n"
        c += f"    if(bucket > {self.input1.num_buckets - 1}) {{\n"
        c += f"        bucket = {self.input1.num_buckets - 1};\n"
        c += f"        input = 1.0;\n"
        c +=  "    }\n"
        if self.input1.minimum != 0.0:
            c += f"    else if(bucket < 0) {{\n"
            c += f"        bucket = 0;\n"
            c += f"        input = 0.0;\n"
            c +=  "    }\n"
        c +=  "    else {\n"
        c +=  "        input = input - bucket;\n"
        c +=  "    }\n"
        c += f"    tbl_ptr += bucket * {self.num_states**2 * (self.order + 1)};\n"
        c +=  "    // Compute the exponential terms of the polynomial.\n"
        for term in range(1, self.order + 1):
            c += f"    const real value{term} = {'*'.join('input' for _ in range(term))};\n"
        c +=  "\n"
        c += f"    real scratch[{self.num_states}] = {{{', '.join('0.0' for _ in range(self.num_states))}}};\n"
        c += f"    for(int col = 0; col < {self.num_states}; ++col) {{\n"
        c +=  "        const real s = *state[col];\n"
        c += f"        for(int row = 0; row < {self.num_states}; ++row) {{\n"
        c +=  "            // Approximate this entry of the matrix.\n"
        for term in range(self.order + 1):
            if term == 0:
                c += f"            const real polynomial = (*tbl_ptr++)"
            else:
                c += f" + (*tbl_ptr++)*value{term}"
        c +=  ";\n"
        c +=  "            scratch[row] += polynomial * s; // Compute the dot product. \n"
        c +=  "        }\n"
        c +=  "    }\n"
        if self.conserve_sum is not None:
            c +=  "    // Conserve the sum of the states.\n"
            c +=  "    real sum_states = 0.0;\n"
            c += f"    for(int x = 0; x < {self.num_states}; ++x) {{ sum_states += scratch[x]; }}\n"
            c += f"    const real correction_factor = {self.conserve_sum} / sum_states;\n"
            c += f"    for(int x = 0; x < {self.num_states}; ++x) {{ scratch[x] *= correction_factor; }}\n"
        c +=  "    // Move the results into the state arrays.\n"
        c += f"    for(int x = 0; x < {self.num_states}; ++x) {{ *state[x] = scratch[x]; }}\n"
        c +=  "}\n\n"
        return c

    def write(self, filename=None):
        if filename is None or filename is True:
            if self.target == 'cuda':
                self.filename = self.name + '.cu'
            elif self.target == 'host':
                self.filename = self.name + '.cpp'
        else:
            self.filename = str(filename)
        self.filename = os.path.abspath(self.filename)
        with open(self.filename, 'wt') as f:
            f.write(self.source_code)
            f.flush()

    def load(self):
        if self._cached_load is not None: return self._cached_load
        scope = {"numpy": np}
        fn_name  = self.name + "_advance"
        inp_name = self.input1.name
        idx_name = self.input1.name + "_indices"
        real_t   = "numpy." + self.float_dtype.__name__
        index_t  = "numpy.int32"
        pycode  = f"def {fn_name}(n_inst, {inp_name}, {idx_name}, "
        pycode +=  ", ".join(self.state_names)
        pycode +=  "):\n"
        pycode +=  "    n_inst = int(n_inst)\n"
        if self.target == 'host':
            for array in [inp_name, idx_name] + self.state_names:
                pycode += f"    assert isinstance({array}, numpy.ndarray), 'isinstance({array}, numpy.ndarray)'\n"
        pycode += f"    assert len({inp_name}) >= n_inst, 'len({inp_name}) >= n_inst'\n"
        pycode += f"    assert {inp_name}.dtype == {real_t}, '{inp_name}.dtype == {real_t}'\n"
        pycode += f"    assert len({idx_name}) == n_inst, 'len({idx_name}) == n_inst'\n"
        pycode += f"    assert {idx_name}.dtype == {index_t}, '{idx_name}.dtype == {index_t}'\n"
        for state in self.state_names:
            pycode += f"    assert len({state}) == n_inst, 'len({state}) == n_inst'\n"
            pycode += f"    assert {state}.dtype == {real_t}, '{state}.dtype == {real_t}'\n"
        args_list = ', '.join(["n_inst", inp_name, idx_name] + self.state_names)
        if self.target == 'host':
            pycode += f"    _entrypoint({args_list})\n"
            scope["_entrypoint"] = self._load_entrypoint_host()
        elif self.target == 'cuda':
            pycode += f"    threads = 32\n"
            pycode += f"    blocks = (n_inst + (threads - 1)) // threads\n"
            pycode += f"    _entrypoint((blocks,), (threads,), ({args_list}))\n"
            scope["_entrypoint"] = self._load_entrypoint_cuda()
        exec(pycode, scope)
        self._cached_load = fn = scope[fn_name]
        return fn

    def _load_entrypoint_host(self):
        src_file = tempfile.NamedTemporaryFile(prefix=self.name+'_', suffix='.cpp', delete=False)
        so_file  = tempfile.NamedTemporaryFile(prefix=self.name+'_', suffix='.so', delete=False)
        src_file.close(); so_file.close()
        self.write(src_file.name)
        subprocess.run(["g++", src_file.name, "-o", so_file.name,
                        "-shared", "-O3"],
                        check=True)
        so = ctypes.CDLL(so_file.name)
        fn = so[self.name + "_advance"]
        argtypes = [ctypes.c_int]
        argtypes.append(np.ctypeslib.ndpointer(dtype=self.float_dtype, ndim=1, flags='C'))
        argtypes.append(np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C'))
        for _ in range(self.num_states):
            argtypes.append(np.ctypeslib.ndpointer(dtype=self.float_dtype, ndim=1, flags='C'))
        fn.argtypes = argtypes
        fn.restype = None
        return fn

    def _load_entrypoint_cuda(self):
        import cupy
        fn_name  = self.name + "_advance"
        module = cupy.RawModule(code=self.source_code,
                                name_expressions=[fn_name],
                                options=('--std=c++11',),)
        return module.get_function(fn_name)
