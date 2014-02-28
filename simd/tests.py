__author__ = 'samyvilar'

import numpy

from shared import zeros
from ctypes_info import to_numpy_type
from simd.ufuncs import *


def test_native_intrinsics(test_size=10**5):
    for operation_name, intrinsic_name in ifilter(lambda item: item[0] in dir(numpy), available_operations.iteritems()):
        all_types = all_intrinsics[intrinsic_name][largest_native_arch]
        for c_type_name in all_types.iterkeys():
            number_of_operands = len(all_types[c_type_name]['operand_type']) + 1
            dtype = to_numpy_type[c_type_name]
            operands = map(zeros, repeat(test_size, number_of_operands), repeat(dtype, number_of_operands))
            for operand in operands:
                operand[:] = (numpy.random.random(test_size) * 100).astype(dtype)
            _ = globals()[operation_name](*operands)
            try:
                if numpy.all(getattr(numpy, operation_name)(*operands) != operands[-1]):
                    raise ValueError('{i} failed!'.format(operation_name))
            except Exception as ex:
                print ex

