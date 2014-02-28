__author__ = 'samyvilar'

import sys
import os
from collections import OrderedDict, deque
from string import lowercase, letters, digits
from itertools import ifilter, ifilterfalse, izip, imap, chain, count, repeat, tee
from multiprocessing import cpu_count, Process
from ctypes import addressof

from simd.intrinsics import all_natively_supported_compilers, all_natively_supported_intrinsics
from simd.intrinsics import all_intrinsics, largest_native_arch, operations_to_intrinsics
from simd.intrinsics import get_required_alignment_in_bytes, family_archs, user_defined_intrinsics_code
from simd.intrinsics import all_natively_supported_headers, all_natively_supported_switches
from ctypes_info import check_attributes, c_type_sizes, to_c_types, as_ctypes
from compile import compile_and_load
from shared import segment_array_into_ctypes


segment_count = cpu_count()

alignment = family_archs.get(
    largest_native_arch, {'required_alignment': {'bytes': None}})['required_alignment']['bytes']


def is_aligned(addr, arch=largest_native_arch):
    return not addr % get_required_alignment_in_bytes(arch)


def exhaust(iterator):
    _ = deque(iterator, maxlen=0)


def check_constant_value(value, max_bit_size=max(i['bits'] for i in c_type_sizes.itervalues())):
    assert isinstance(value, (int, long, float)) and getattr(value, 'bit_length', lambda: 0)() < max_bit_size
    return 1


def number_of_elements(v, attrs=(
    ('size', lambda n: n.size),  # numpy types ...
    ('capitalize', len),  # str types otherwise infinite recursion ...
    ('_type_', lambda ct: sum(imap(number_of_elements, ct)) if hasattr(ct._type_, '__len__') else len(ct)),
    ('__len__', lambda r: sum(imap(number_of_elements, r))),  # container types recursively call :( (slow!!!)
    ('__sizeof__', check_constant_value),  # virtually all other types, check they are numeric and in range ...
)):
    return check_attributes(v, attrs, attrs[-1])


def error_if_not_type(v, types):
    assert isinstance(v, types)
    return v


def get_value(v):
    return error_if_not_type(get_value(v[0]) if hasattr(v, '__getitem__') else v, (int, long, float))


def get_c_type(a, to_c_types=to_c_types, attrs=(
    ('dtype', lambda n, to_c_types=to_c_types: to_c_types[n.dtype]),  # numpy ...
    ('_type_', lambda n, to_c_types=to_c_types: to_c_types[n._type_]),  # ctypes
    ('__len__', lambda n, to_c_types=to_c_types: to_c_types[type(get_value(n))]),  # container types ...
    ('__sizeof__', lambda n, to_c_types=to_c_types: to_c_types[type(n)]),  # all other types ...
)):
    return check_attributes(a, attrs, attrs[-1])


                                            # numpy arrays                       ctypes array
def get_address(v, arch=largest_native_arch, methods=(
    ('ctypes', lambda v: v.ctypes.data), ('_type_', lambda v: addressof(v))
)):
    return check_attributes(v, methods)


def is_address_aligned(addr, alginment):
    return not (addr % alignment)


def get_move_intrinsic(addr, arch, all_intrinsics, instr_name):
    return get_alternative_if_supported(all_intrinsics[instr_name][arch][
        'aligned' if is_address_aligned(addr, get_required_alignment_in_bytes(arch)) else 'unaligned'
    ])


def get_alternative_if_supported(instrs):
    alternative = instrs.get('alternative', instrs)['name']
    return alternative if alternative in all_natively_supported_intrinsics else instrs['name']


def store_intrinsic(addr=None, arch=largest_native_arch, all_intrinsics=all_intrinsics):
    return get_move_intrinsic(addr, arch, all_intrinsics, 'store')


def load_intrinsic(addr=None, arch=largest_native_arch, all_intrinsics=all_intrinsics):
    return get_move_intrinsic(addr, arch, all_intrinsics, 'load')


def initialization_code(
        a,
        to_type=None,
        a_name='a',
        vector_name='vector_a',
        arch=largest_native_arch,
        all_intrinsics=all_intrinsics
):
    operand_type = to_type or get_c_type(a)
    v_type = all_intrinsics['set1'][arch][to_type]['return_type']
    if number_of_elements(a) == 1:  # if scalar check value in range then type cast and insert into code as constant ...
        init_func = all_intrinsics['set1'][arch][operand_type]['name']  # load scalar for broadcasting ...
        value = '({t}){v}'.format(t=all_intrinsics['set1'][arch][operand_type]['operand_type'][0], v=get_value(a))
    else:  # if vector simply type cast to appropriate vector type pointer ...
        v_type += ' *'
        init_func = '(' + v_type + ')'
        value = a_name
    return '{v_type} {v_name} = {init_func}({value});'.format(
        v_type=v_type, v_name=vector_name, init_func=init_func, value=value
    )


def operation_code(var_names, oper, operand_types=None, arch=largest_native_arch, all_intrinsics=all_intrinsics):
    return '{oper_func}({names})'.format(
        oper_func=all_intrinsics[oper][arch][operand_types]['name'],
        names=','.join(var_names)
    )


def vectorized_func(
        operation,
        operands,
        func_name='',
        ops_per_loop=1,
        arch=largest_native_arch,
        all_intrinsics=all_intrinsics,
        user_defined_instrs_code=user_defined_intrinsics_code
):
    hash_operand = lambda o: (get_value if number_of_elements(o) == 1 else id)(o)
    unique_operns = OrderedDict(
        OrderedDict(  # give each 'unique' operand a unique name to reduce the number of registers in use ...
            izip(     # if a single value just use that value as the id otherwise we need to get object ids ...
                imap(hash_operand, operands),
                izip(((c + str(pf)) for pf in chain(('',), count()) for c in lowercase), operands)
            )
        ).itervalues()
    )
    identifiers = dict(izip(imap(hash_operand, unique_operns.itervalues()), unique_operns.iterkeys()))
    get_operand_identifier = lambda operand, identifiers=identifiers: identifiers[hash_operand(operand)]

    unique_scalars = OrderedDict(ifilter(lambda pr: number_of_elements(pr[1]) == 1, unique_operns.iteritems()))
    unique_vectors = OrderedDict(ifilterfalse(lambda pr: pr[0] in unique_scalars, unique_operns.iteritems()))

    # check that vectors are properly aligned ...
    assert all(imap(is_aligned, imap(get_address, chain(unique_vectors.itervalues()))))
    # check that vectors are the same length if any
    assert len(set(imap(number_of_elements, chain(unique_vectors.itervalues())))) <= 1
    # check that vectors are the same type if any
    # assert len(set(imap(get_c_type, chain(unique_vectors.itervalues())))) <= 1

    out_name = next(reversed(unique_operns))
    out = unique_operns[out_name]
    c_type = get_c_type(out)
    number_of_operations, rem = divmod(
        len(out), (family_archs[arch]['vector_size']['bytes']/c_type_sizes[get_c_type(out)]['bytes'])
    )
    assert not rem

    initialization = os.linesep.join(  # initialize scalars, vectors will be passed as parameters ...
        (initialization_code(vl, c_type, nm, nm, arch, all_intrinsics) for nm, vl in unique_scalars.iteritems())
    )

    single_operation = '{store}({o}{upd}, {oper}){upd_out};'.format(
        store=store_intrinsic(get_address(out)),
        o=out_name,
        upd='++' if out_name not in unique_vectors else '',  # update vector if its not being referenced in operations.
        upd_out=',{o}++'.format(o=out_name) if out_name in unique_vectors else '',  # update vector outside if ref
        oper=operation_code(
            ('{load_or_ref}({v})'.format(
                load_or_ref=load_intrinsic(get_address(unique_vectors[o]), arch, all_intrinsics) if o in unique_vectors else '',
                v=o + ('++' if o in unique_vectors and o != out_name else '')  # increment non-out vectors ...
            ) for o in imap(get_operand_identifier, operands[:-1])),
            operation,
            c_type
        )
    ) + os.linesep

    loop = """
    unsigned {t} count = {l};
    while (count--)
    {{
        {oper}
    }}
    {rem}
    """

    if None:  # ops_per_loop >= number_of_operations:
        loop_code = single_operation * number_of_operations  # unroll loop completely no need for while statement ...
    else:
        ops_per_loop = 1
        length, rem = divmod(number_of_operations, ops_per_loop)
        loop_code = loop.format(
            t=('long long' if length >= 2**32 else '') + 'int',  # use faster int if value small enough ...
            l=length,
            oper=single_operation * ops_per_loop,
            rem=(single_operation * rem)
        )

    return '''
{headers}

{user_code}
void {func_name} ({parameters})
{{
    {initialization}
    {loop}
}}
    '''.format(
        headers=''.join(imap('#include {0}{1}'.format, all_natively_supported_headers, repeat(os.linesep))),
        func_name=func_name or 'vectorized_{o}_{c}_{t}s'.format(
            o=operation, l=len(operation) - 1, c=number_of_elements(out), t=c_type.replace(' ', '_')
        ),
        parameters=', '.join(
            imap('__m{0}i *'.format(family_archs[arch]['vector_size']['bits']).__add__, unique_vectors.iterkeys())
        ),
        initialization=initialization,
        loop=loop_code,
        user_code=user_defined_instrs_code
    ), imap(as_ctypes, unique_vectors.itervalues())


valid_identifier_chars = set(letters + digits + '_')
invalid_identifier_chars = set(imap(chr, xrange(256))) - valid_identifier_chars
to_valid_identifier = dict(chain(
    izip(valid_identifier_chars, valid_identifier_chars),
    izip(invalid_identifier_chars, repeat('_')))
)

cache = {}


def single_sse_vector_operation(
        _o_,
        cc='gcc',
        arch=largest_native_arch,
        all_intrinsics=all_intrinsics,
        to_valid_identifier=to_valid_identifier,
        user_defined_instrs_code=user_defined_intrinsics_code,
        cache=cache
):
    def func(*operands):
        func_name = ''.join(imap(
            to_valid_identifier.__getitem__,
            'vectorized_{a}_{o}_{t}_{l}'.format(a=arch, o=_o_, t=get_c_type(operands[-1]), l=len(operands[-1]))
        ))
        code, func_args = vectorized_func(
            _o_, operands, func_name=func_name, ops_per_loop=1000, arch=arch, all_intrinsics=all_intrinsics,
            user_defined_instrs_code=user_defined_instrs_code
        )
        if code not in cache:
            mod = compile_and_load(
                code,
                compiler=cc,
                extra_compile_args=chain(('-Ofast',), all_natively_supported_switches[cc])
            )
            cache[code] = getattr(mod, func_name)
        func_ptr = cache[code]
        f_args, f_types = tee(func_args)
        func_ptr.argtypes, func_ptr.restype = tuple(imap(type, f_types)), None
        func_ptr(*f_args)
        return operands[-1]
    return func


def apply_func(func, operands):
    p = Process(target=func, args=operands)
    p.start()
    return p


def parallelized_func(operands, func_impl, segment_count=segment_count):
    segmented_operands = izip(*imap(
        lambda v, number_of_cores:
        # if vector is a single value than broadcast it out, either-wise segment it ...
        repeat(v) if number_of_elements(v) == 1 else segment_array_into_ctypes(v, number_of_cores),
        operands,
        repeat(segment_count),
    ))
    start_processes, join_processes = tee(imap(apply_func, repeat(func_impl), segmented_operands))
    _, _ = exhaust(start_processes), exhaust(imap(lambda p: p.join(), join_processes))


available_operations = operations_to_intrinsics
if all_natively_supported_compilers:  # add operations to module if we have at least one compiler ...
    for operation_name, intrinsic_name in operations_to_intrinsics.iteritems():
        setattr(
            sys.modules[__name__],
            operation_name,
            single_sse_vector_operation(
                intrinsic_name,
                all_natively_supported_compilers[0],
                largest_native_arch,
                all_intrinsics=all_intrinsics,
                to_valid_identifier=to_valid_identifier,
                user_defined_instrs_code=user_defined_intrinsics_code,
            )
        )
