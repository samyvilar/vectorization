__author__ = 'samyvilar'

import timeit
from string import lowercase
from itertools import product, repeat, imap, izip, chain, count
from intrinsics import all_natively_supported_switches
from collections import defaultdict, deque

import numpy

from shared import zeros
from ctypes_info import byte_bit_size, as_ctypes
from compile import compile_and_load
from intrinsics import vectorized_func


def locate_optimal_ops_per_loop(
        values_to_try=xrange(1, 257),
        test_vector_size=256,
        dtypes=('float32',),
        operations=('add',),
        timeit_number=10,
        timeit_repeat=5,
        compiler='gcc'
):
    stats = defaultdict(dict)
    for dt, op_name in product(dtypes, operations):
        a, b, out = imap(zeros, repeat(test_vector_size, 3), repeat(dt))
        operands = a, b, out
        a[:], b[:], out[:] = (
            numpy.random.random_sample(out.shape) if 'float' in dt
            else numpy.random.random_integers(
                -(2 ** ((out.itemsize - 1) * byte_bit_size)), (2 ** ((out.itemsize - 1) * byte_bit_size)) - 1,
                size=out.shape
            ) for _ in xrange(3))
        ac, bc, outc = imap(as_ctypes, operands)
        funcs = tuple(
            getattr(
                compile_and_load(
                    vectorized_func(
                        op_name,
                        izip(((c + str(pf)) for pf in chain(('',), count()) for c in lowercase), operands),
                        func_name='vectorized_add_{l}_{t}'.format(l=ops_per_loop, t=dt),
                        ops_per_loop=ops_per_loop
                    )[0],
                    compiler=compiler,
                    extra_compile_args=chain(('-Ofast',), all_natively_supported_switches[cc]),
                    silent=True
                ),
                'vectorized_add_{l}_{t}'.format(l=ops_per_loop, t=dt),
            ) for ops_per_loop in values_to_try)
        _ = deque(imap(setattr, funcs, repeat('restype'), repeat(None)), 0)
        timings = [timeit.repeat(
            lambda f=f, ac=ac, bc=bc, outc=outc: f(ac, bc, outc), number=timeit_number, repeat=timeit_repeat
        ) for f in funcs]
        stats[op_name][dt] = dict(izip(
            ('min', 'avg', 'timings'),
            chain(
                (1 + s for s in imap(numpy.argmin, (numpy.min(timings, axis=1), numpy.average(timings, axis=1)))),
                (timings,)
            )
        ))
    return stats



# >>> stats = locate_optimal_ops_per_loop(dtypes=('float64', 'float32', 'int64', 'int32', 'int16', 'int8'))
# >>> stats['add']['int8']['min']
# 1
# >>> stats['add']['int16']['min']
# 1
# >>> stats['add']['int32']['min']
# 2
# >>> stats['add']['int64']['min']
# 16
# >>> stats['add']['float32']['min']
# 4
# >>> stats['add']['float64']['min']
# 1
# >>>


# >>> stats = locate_optimal_ops_per_loop(
#   test_vector_size=10**5, dtypes=('float64', 'float32', 'int64', 'int32', 'int16', 'int8')
# )
# >>> stats['add']['int8']['min']
# 13
# >>> stats['add']['int16']['min']
# 47
# >>> stats['add']['int32']['min']
# 52
# >>> stats['add']['int64']['min']
# 2
# >>> stats['add']['float32']['min']
# 76
# >>> stats['add']['float64']['min']
# 120
# >>>

# >>> stats = locate_optimal_ops_per_loop(
#   test_vector_size=10**6, dtypes=('float64', 'float32', 'int64', 'int32', 'int16', 'int8')
# )
# >>> stats['add']['int8']['min']
# 52
# >>> stats['add']['int16']['min']
# 42
# >>> stats['add']['int32']['min']
# 54
# >>> stats['add']['int64']['min']
# 4
# >>> stats['add']['float32']['min']
# 47
# >>> stats['add']['float64']['min']
# 17
# >>>


# stats = locate_optimal_ops_per_loop(
#   test_vector_size=10**6, dtypes=('float64', 'float32', 'int64', 'int32', 'int16', 'int8'), compiler='icc'
# )
# >>> stats['add']['int8']['min']
# 148
# >>> stats['add']['int16']['min']
# 161
# >>> stats['add']['int32']['min']
# 13
# >>> stats['add']['int64']['min']
# 20
# >>> stats['add']['float32']['min']
# 19
# >>> stats['add']['float64']['min']
# 13
# >>>


# >>> stats = locate_optimal_ops_per_loop(
#   dtypes=('float64', 'float32', 'int64', 'int32', 'int16', 'int8'), compiler='icc'
# )
# >>> stats['add']['int8']['min']
# 1
# >>> stats['add']['int16']['min']
# 42
# >>> stats['add']['int32']['min']
# 8
# >>> stats['add']['int64']['min']
# 1
# >>> stats['add']['float32']['min']
# 80
# >>> stats['add']['float64']['min']
# 54
# >>>

# >>> stats = locate_optimal_ops_per_loop(dtypes=('float64', 'float32', 'int32', 'int16'), operations=('mul',))
# >>> stats['mul']['int16']['min']
# 6
# >>> stats['mul']['int32']['min']
# 1
# >>> stats['mul']['float32']['min']
# 1
# >>> stats['mul']['float64']['min']
# 1
# >>>

