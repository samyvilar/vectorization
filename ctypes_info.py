__author__ = 'samyvilar'

import numpy
from itertools import imap, chain, repeat, izip, tee, ifilter
from ctypes import sizeof
import ctypes


# C types: 'char', 'int', ..., numpy types 'float32', 'float64', ctypes: ctypes.c_short, python types: int, float
byte_bit_size = 8
c_integral_types, c_real_types = ('char', 'short', 'int', 'long', 'long long'), ('float', 'double')
c_unsigned_types = tuple(imap('unsigned '.__add__, c_integral_types))
c_types_types = 'integral', 'unsigned', 'real'
numeric_types = {t: globals()['c_' + t + '_types'] for t in c_types_types}
iterate_all_c_types = lambda numeric_types=numeric_types: chain.from_iterable(numeric_types.itervalues())

_c_type_to_ctypes = lambda ct: ('c_' + ct).replace(' ', '').replace('unsigned', 'u').replace('uchar', 'ubyte')
ctypes_type_to_c_type = dict(chain.from_iterable(
    imap(
        lambda f, v: f(v),
        (
            lambda names: imap(lambda v: (_c_type_to_ctypes(v), v), names),  # strs ...
            lambda names: imap(lambda o: (getattr(ctypes, _c_type_to_ctypes(o)), o), names),  # obj types ...
        ),
        tee(iterate_all_c_types())
    )
))

c_type_sizes = dict(imap(
    lambda ct: (ct, {
        'bytes': sizeof(getattr(ctypes, _c_type_to_ctypes(ct))),
        'bits': byte_bit_size * sizeof(getattr(ctypes, _c_type_to_ctypes(ct)))
    }), iterate_all_c_types()
))

numpy_type_to_c_type = dict(
    chain.from_iterable(
        imap(
            lambda nt, prefix:
            chain.from_iterable(
                imap(
                    lambda entry: (entry, (numpy.dtype(entry[0]), entry[1])),
                    (
                        ('{0}{1}'.format(prefix, size_info['bits']), cn)
                        for cn, size_info in izip(numeric_types[nt], imap(c_type_sizes.__getitem__, numeric_types[nt]))
                    )
                )
            ),
            numeric_types,
            imap({'integral': 'int', 'unsigned': 'uint', 'real': 'float'}.__getitem__, numeric_types)
        )
    )
)

python_type_to_c_type = {
    int: 'long long',
    'int': 'long long',
    float: 'double',
    'float': 'double',
    long: 'long long',
    'long': 'long long'
}


to_c_types = dict(
    chain.from_iterable(
        tps.iteritems()
        for tps in imap(
            globals().__getitem__,
            ifilter(lambda v: v.endswith('_type_to_c_type') and isinstance(globals()[v], dict), globals().iterkeys())
        )
    )
)


numpy_type_to_ctypes = dict(
    izip(
        numpy_type_to_c_type.iterkeys(),
        imap(getattr, repeat(ctypes), imap(_c_type_to_ctypes, numpy_type_to_c_type.itervalues()))
    )
)

as_ctypes = lambda vector, methods=(
    ('ctypes', numpy.ctypeslib.as_ctypes),  # numpy ...
    ('_type_', lambda v: v),  # ctypes do nothing ...
): check_attributes(vector, methods)

to_numpy_type = dict(
    chain(
        imap(reversed, numpy_type_to_c_type.iteritems()),
        izip(numpy_type_to_ctypes.iterkeys(), numpy_type_to_ctypes.iterkeys()),
        imap(reversed, numpy_type_to_ctypes.iteritems())
    )
)
to_numpy_type['long'] = to_numpy_type['long long'] if c_type_sizes['long'] == c_type_sizes['long long'] \
    else to_numpy_type['int']
to_numpy_type['unsigned long'] = to_numpy_type['unsigned long long'] \
    if c_type_sizes['unsigned long'] == c_type_sizes['unsigned long long']\
    else to_numpy_type['unsigned int']

__none__ = object()


def check_attributes(v, methods, default=__none__):  # though it is slower ...
    # filter out all those attributes not present within the object in question...
    impl = next(ifilter(lambda attr, v=v: hasattr(v, attr[0]), methods), default)
    if impl is __none__:  # the object did not contain any of the attributes, check if a default entry was supplied ...
        raise AttributeError
    return impl[1](v)  # otherwise call function of located attribute name with value for appropriate action ...
