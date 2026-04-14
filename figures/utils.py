from pathlib import Path
import glob
import importlib
import os
import re
import shutil
import subprocess
import tempfile

def get_solver(nmodl_text):
    for match in re.finditer(r"\sMETHOD\s+(\w+)\b", nmodl_text):
        return match.groups()[0]
    for match in re.finditer(r"\sSTEADYSTATE\s+(\w+)\b", nmodl_text):
        return match.groups()[0]

def set_solver(mod_dir, method):
    # Fixup each file
    for path in Path(mod_dir).glob("*.mod"):
        with open(path, 'rt') as f:
            text = f.read()
        # 
        if get_solver(text) in ["cnexp"]:
            continue
        # Replace the solver method
        if not method.startswith("approx"):
            text = re.sub(r"\sMETHOD\s+\w+\s", f" METHOD {method}\n", text)
            text = re.sub(r"\sSTEADYSTATE\s+\w+\s", f" STEADYSTATE {method}\n", text)
        # 
        with open(path, 'wt') as f:
            f.write(text)

def all_mod_files():
    fig_dir = Path(__file__).parent
    mod_dir = fig_dir.parent.joinpath("mod")
    mod_files = [fig_dir / "presyn.mod"]
    mod_files.extend(mod_dir.glob("*.mod"))
    return mod_files

def dedup_mod_files():
    fig_dir = Path(__file__).parent
    mod_dir = fig_dir.parent.joinpath("mod")
    mod_files = [fig_dir / "presyn.mod"]
    mod_files.append(mod_dir / "AMPA_13state.mod")
    mod_files.append(mod_dir / "NMDA_10state.mod")
    mod_files.append(mod_dir / "Nav11_6state.mod")
    mod_files.append(mod_dir / "Kv11_4state.mod")
    mod_files.append(mod_dir / "Kv11_6state.mod")
    mod_files.append(mod_dir / "Kv11_11state.mod")
    mod_files.append(mod_dir / "Kv11_13state.mod")
    return mod_files

def copy_mod_files(mod_files):
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
    return out_dir

def load(mod_files, method, dt=None, c=None, error=None):
    global neuron
    # Locate the argument NMODL files and copy them to a temporary directory for processing.
    out_dir = copy_mod_files(mod_files)
    # 
    set_solver(out_dir, method)
    # Build the approximation
    if method.startswith("approx"):
        in_dir = out_dir
        out_dir = Path(tempfile.mkdtemp())
        presyn = in_dir.joinpath("presyn.mod")
        if presyn.exists():
            presyn.rename(out_dir.joinpath("presyn.mod"))
        cmd = ["matexp", "-v", "-v", "-dt", str(dt), "-t", str(c)]
        if error:
            cmd.extend(["-e", str(error)])
        if method.endswith("32"):
            cmd.extend(["-f", "32"])
        cmd.extend(["--input", "v", "-100", "100"])
        cmd.extend(["--input", "C", "0", "10"])
        cmd.extend(["--log", "C"])
        for in_path in in_dir.iterdir():
            subprocess.run(cmd + [in_path, out_dir], check=True)
    # Compile the NMODL files
    return build_models(out_dir, method in ["matexp"])

def build_models(mod_dir, nmodl):
    cwd = os.getcwd()
    os.chdir(mod_dir)
    print("MOD DIR:", mod_dir)
    # Compile the NMODL files
    build_command = ["nrnbuild.py", "--parallel", "7", "--verbose"]
    if nmodl:
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
