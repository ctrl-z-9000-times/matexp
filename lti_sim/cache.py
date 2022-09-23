"""
The cache is a pickled file structured as a tuple of (checksum, solutions)
Where solutions is a dictionary of arguments -> source-code

If checksum does not match or if the NMODL file was modified more recently than
the cache file then the cache is deleted and reinitialized.
"""
import hashlib
import os
import pickle

def _get_cache_filename(nmodl_filename):
    nmodl_dirname, nmodl_filename = os.path.split(nmodl_filename)
    nmodl_filename, ext = os.path.splitext(nmodl_filename)
    return os.path.join(nmodl_dirname, '.' + nmodl_filename + '.cache')

def _get_arguments_key(arguments):
    return ','.join(repr(arg) for arg in arguments)

def _checksum_nmodl_file(nmodl_filename):
    with open(nmodl_filename, 'rt') as nmodl_file:
        nmodl_text = nmodl_file.read()
    return hashlib.md5(nmodl_text.encode()).hexdigest()

def _open_cache(nmodl_filename, cache_filename):
    """ Assumes that the cache exists, but might be stale. """
    if os.path.getmtime(nmodl_filename) > os.path.getmtime(cache_filename):
        return None
    with open(cache_filename, 'rb') as cache_file:
        cache = pickle.load(cache_file)
    if cache[0] != _checksum_nmodl_file(nmodl_filename):
        return None
    return cache

def read(nmodl_filename, arguments):
    cache_filename = _get_cache_filename(nmodl_filename)
    if not os.path.exists(cache_filename):
        return None
    cache = _open_cache(nmodl_filename, cache_filename)
    if cache is None:
        os.remove(cache_filename)
        return None
    return cache[1].get(_get_arguments_key(arguments), None)

def write(nmodl_filename, arguments, source_code):
    cache_filename = _get_cache_filename(nmodl_filename)
    if os.path.exists(cache_filename):
        cache = _open_cache(nmodl_filename, cache_filename)
        assert cache is not None, "NMODL file changed while running!"
    else:
        cache = (_checksum_nmodl_file(nmodl_filename), {})
    cache[1][_get_arguments_key(arguments)] = source_code
    with open(cache_filename, 'wb') as cache_file:
        pickle.dump(cache, cache_file)
