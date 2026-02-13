from pathlib import Path
import glob
import importlib
import os
import re
import shutil
import subprocess
import tempfile

def load(mod_files, method, dt=None, c=None, error=None):
    global neuron
    # Locate the argument NMODL files and copy them to a temporary directory for processing.
    out_dir = Path(tempfile.mkdtemp())
    for path in mod_files:
        path = Path(path).resolve()
        if path.is_file():
            shutil.copy2(path, out_dir)
        elif path.is_dir():
            for p in path.glob("*.mod"):
                shutil.copy2(p, out_dir)
        else:
            raise ValueError("No such file", path)
    # Fixup each file
    for path in out_dir.iterdir():
        with open(path, 'rt') as f:
            text = f.read()
        # Replace the solver method
        if method != "approx":
            text = re.sub(r"\sMETHOD\s+\w+\s", f" METHOD {method}\n", text)
            text = re.sub(r"\sSTEADYSTATE\s+\w+\s", f" STEADYSTATE {method}\n", text)
        # 
        with open(path, 'wt') as f:
            f.write(text)
    # Build the approximation
    if method == "approx":
        in_dir = out_dir
        out_dir = Path(tempfile.mkdtemp())
        presyn = in_dir.joinpath("presyn.mod")
        if presyn.exists():
            presyn.rename(out_dir.joinpath("presyn.mod"))
        cmd = ["matexp", "-v", "-v", "-t", str(dt), "-c", str(c)]
        if error:
            cmd.extend(["-e", str(error)])
        cmd.extend(["--input", "v", "-100", "100"])
        cmd.extend(["--input", "C", "0", "10"])
        for in_path in in_dir.iterdir():
            subprocess.run(cmd + [in_path, out_dir], check=True)
    # 
    cwd = os.getcwd()
    os.chdir(out_dir)
    print("MOD DIR:", out_dir)
    # Compile the NMODL files
    build_command = ["nrnbuild.py", "--parallel", "7", "--verbose"]
    if method in ["matexp"]:
        build_command.append("--nmodl")
    else:
        pass # Use nocmodl
    subprocess.run(build_command, check=True)
    # Load NEURON
    neuron = importlib.import_module("neuron")
    from neuron import n, gui
    # 
    os.chdir(cwd)
    return neuron

def mechanism_names(mod_files):
    import neuron.nmodl
    import neuron.nmodl.dsl
    from neuron.nmodl.dsl import symtab, visitor
    from neuron.nmodl.symtab import NmodlType
    AstNodeType = neuron.nmodl.dsl.ast.AstNodeType

    mechanism_data = {}
    for path in mod_files:
        with open(path, 'rt') as file:
            text = file.read()
        driver = neuron.nmodl.NmodlDriver()
        ast = driver.parse_string(text)
        lookup_v = visitor.AstLookupVisitor()
        suffix = lookup_v.lookup(ast, AstNodeType.SUFFIX)
        name = suffix[0].name.get_node_name()
        sym_v = symtab.SymtabVisitor()
        sym_v.visit_program(ast)
        table = ast.get_symbol_table()
        states = table.get_variables_with_properties(NmodlType.state_var)
        state_names = sorted(token.get_name() for token in states)
        mechanism_data[name] = state_names
    return mechanism_data
