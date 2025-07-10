import subprocess
import os

def test_nav11():
    directory   = os.path.dirname(__file__)
    test_file   = os.path.join(directory, 'Nav11.mod')
    output_file = os.path.join(directory, 'Nav11_tmp.mod')
    if os.path.exists(output_file): os.remove(output_file)
    subprocess.run(['python', '-m', 'matexp', # Test calling program as a python module.
            test_file, output_file, '-i', 'v', '-120', '120',
            '-t', '0.1', '-c', '37',
            '--verbose'],
            check=True,)
    with open(output_file, 'rt') as f:
        assert len(f.read()) > 100 # Check file is not empty.
    os.remove(output_file)

def test_ampa():
    directory   = os.path.dirname(__file__)
    test_file   = os.path.join(directory, 'ampa13.mod')
    output_file = os.path.join(directory, 'ampa13_tmp.mod')
    if os.path.exists(output_file): os.remove(output_file)
    subprocess.run(['matexp', # Test calling program as a CLI script.
            test_file, output_file,
            '-i', 'C', '0', '1e3', '--log',
            '-t', '0.1', '-c', '37',
            # '--target', 'cuda',
            '-f32',
            '--verbose'],
            check=True,)
    with open(output_file, 'rt') as f:
        assert len(f.read()) > 100 # Check file is not empty.
    os.remove(output_file)

def test_nmda():
    directory   = os.path.dirname(__file__)
    test_file   = os.path.join(directory, 'NMDA.mod')
    output_file = os.path.join(directory, 'NMDA_tmp.mod')
    if os.path.exists(output_file): os.remove(output_file)
    subprocess.run(['python', '-m', 'matexp',
            test_file, output_file,
            '-i', 'C', '0', '1e3', '--log',
            # Test default input "v"
            '-t', '0.1',
            # '--target', 'cuda',
            '-f32',
            '-e', '1e-3',
            '-vv'],
            check=True,)
    with open(output_file, 'rt') as f:
        assert len(f.read()) > 100 # Check file is not empty.
    os.remove(output_file)
