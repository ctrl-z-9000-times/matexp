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

env = dict(os.environ)
env['NOSHOW'] = '1'

for file in mod_files:
	cmd = [program, "--time_step", ".100", file]
	print(' '.join(str(x) for x in cmd))
	run(cmd, env=env, check=True)

os.chdir(cwd)
