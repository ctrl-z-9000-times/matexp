#!/usr/bin/env python

from pathlib import Path
from subprocess import run
import os

figure_dir = Path(__file__).parent 
program = figure_dir / "propagator_matrix.py"
mod_files = figure_dir.glob("../mod/*.mod")
out_dir = figure_dir / "propagator_out"
out_dir.mkdir(exist_ok=True)
cwd = Path.cwd()
print("OUT DIR:", out_dir)
os.chdir(out_dir)

for file in mod_files:
	cmd = [program, "-q", "--time_step", ".100", file]
	print(' '.join(str(x) for x in cmd))
	run(cmd, check=True)

os.chdir(cwd)
