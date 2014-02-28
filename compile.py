__author__ = 'samyvilar'

import subprocess
import platform
import tempfile
import os
import ctypes

from itertools import chain
from collections import defaultdict


def compile_code(code, compiler='gcc', compiler_name='', language='c', extra_compile_args=(), silent=False):
    # compiles a string code into a shared object, using temp files and returns the raw binaries ...
    compiler_name = compiler_name or os.path.basename(compiler)
    shared_flag = defaultdict(lambda: '-shared')
    if getattr(platform, 'mac_ver', lambda: (None,))()[0]:
        shared_flag['Darwin', 'icc'] = '-dynamiclib'  # icc on MacOSX doesn't support -shared must use -dynamiclib
        shared_flag['Darwin', 'nvcc'] = '-shared -ccbin=/usr/bin/clang --compiler-options'
    input_file, output_file = tempfile.NamedTemporaryFile(), tempfile.NamedTemporaryFile()
    input_file.write(code)
    input_file.flush()
    try:
        output = subprocess.check_output(
            list(chain(
                (compiler,
                 shared_flag[platform.system(), compiler_name],
                 '-fPIC',
                 '-x{l}'.format(l=language),
                 input_file.name,
                 '-o',
                 output_file.name),
                extra_compile_args
            )),
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as er:
        print er
        print er.output
        print code
        raise er
    if output and not silent:
        print output
    with open(output_file.name, 'rb') as b_file:
        return b_file.read()


def load_shared_obj(so_bins):
    with tempfile.NamedTemporaryFile() as shared_obj_path:
        shared_obj_path.write(so_bins)
        shared_obj_path.flush()
        return ctypes.CDLL(shared_obj_path.name)


def compile_and_load(code, **kwargs):
    return load_shared_obj(compile_code(code, **kwargs))

