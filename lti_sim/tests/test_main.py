import subprocess
import os

def test_nav11():
    directory   = os.path.dirname(__file__)
    test_file   = os.path.join(directory, 'Nav11.mod')
    output_file = os.path.join(directory, 'tmp.cu')
    if os.path.exists(output_file): os.remove(output_file)
    subprocess.run(['python', '-m', 'lti_sim',
            test_file, '-i', 'v', '-120', '120',
            '-t', '0.1', '-c', '37',
            '--benchmark',
            '-o', output_file])
    with open(output_file, 'rt') as f:
        assert len(f.read()) > 100 # Check file is not empty.
    os.remove(output_file)

def test_ampa():
    directory   = os.path.dirname(__file__)
    test_file   = os.path.join(directory, 'ampa13.mod')
    output_file = os.path.join(directory, 'tmp.cu')
    if os.path.exists(output_file): os.remove(output_file)
    subprocess.run(['python', '-m', 'lti_sim',
            test_file, '-i', 'C', '0', '1e3', '--logarithmic',
            '-t', '0.1', '-c', '37',
            '--benchmark',
            '-o', output_file])
    with open(output_file, 'rt') as f:
        assert len(f.read()) > 100 # Check file is not empty.
    os.remove(output_file)
