from pathlib import Path
import glob
import importlib
import os
import re
import shutil
import subprocess
import tempfile

def load(mod_files, method, zero_conductance, dt=None, c=None, error=None):
    global neuron
    # Locate the argument NMODL files and copy them to a temporary directory for processing.
    out_dir = Path(tempfile.mkdtemp())
    for path in mod_files:
        path = Path(path)
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
        if zero_conductance:
            text = re.sub(r"\sgmax\s*=\s*\d+(\.\d*)?\s", " gmax = 0\n", text)
        with open(path, 'wt') as f:
            f.write(text)
    # Build the approximation
    if method == "approx":
        in_dir = out_dir
        out_dir = Path(tempfile.mkdtemp())
        cmd = ["matexp", "-v", "-v", "-t", str(dt), "-c", str(c)]
        if error:
            cmd.extend(["-e", str(error)])
        cmd.extend(["--input", "v", "-100", "100"])
        for in_path in in_dir.iterdir():
            subprocess.run(cmd + [in_path, out_dir], check=True)
    # 
    cwd = os.getcwd()
    os.chdir(out_dir)
    print("MOD DIR:", out_dir)
    # Compile the NMODL files
    subprocess.run(["nrnivmodl-all-cmake"], check=True)
    # Load NEURON
    neuron = importlib.import_module("neuron")
    from neuron import n, gui
    # 
    os.chdir(cwd)
    return neuron
