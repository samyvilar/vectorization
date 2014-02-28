__author__ = 'samyvilar'

import tempfile
import re
import os

from collections import defaultdict
from itertools import imap, izip, repeat, chain, ifilter, ifilterfalse, product
import numpy

from cpuinfo import does_cpu_support, get_all_cpu_features
from ctypes_info import byte_bit_size, c_type_sizes, numeric_types
from ctypes_info import c_types_types

import subprocess


sse, sse2, sse3, ssse3, sse4_1, sse4_2, sse4a = 'sse', 'sse2', 'sse3', 'ssse3', 'sse4.1', 'sse4.2', 'sse4a'  # names ...
avx, avx2, avx512 = 'avx', 'avx2', 'avx-512'
instruction_sets = {sse, sse2, sse3, ssse3, sse4_1, sse4_2, sse4a, avx, avx2}
if any(imap(lambda f: '512' in f,  get_all_cpu_features())):
    instruction_sets |= {avx512}

common_prefix = '_mm'
common_prefixes = {sse: common_prefix, avx: common_prefix + '256', avx512: common_prefix + '512'}

real_postfixes = {'float': 'ps', 'double': 'pd'}
integral_postfixes = dict((cname, 'epi' + str(c_type_sizes[cname]['bits'])) for cname in numeric_types['integral'])
unsigned_postfixes = dict((cname, 'epu' + str(c_type_sizes[cname]['bits'])) for cname in numeric_types['unsigned'])
postfixes = dict(chain.from_iterable(globals()[cts + '_postfixes'].iteritems() for cts in c_types_types))

family_archs = {
    sse: {
        'required_alignment': {'bytes': 16},
        'vector_size': {'bytes': 16, 'bits': 128},
        'versions': {
            sse: {'headers': ('<xmmintrin.h>',)},
            sse2: {'headers': ('<emmintrin.h>',)},
            sse3: {'headers': ('<pmmintrin.h>',)},
            ssse3: {'headers': ('<tmmintrin.h>',)},
            sse4_1: {'headers': ('<smmintrin.h>',)},
            sse4_2: {'headers': ('<nmmintrin.h>',)},
            sse4a: {'headers': ('<ammintrin.h>',)}
        },
    },
    avx: {
        'required_alignment': {'bytes': 32},
        'vector_size': {'bytes': 32, 'bits': 256},
        'versions': {
            avx: {'headers': ('<immintrin.h>',)},
            avx2: {'headers': ('<immintrin.h>',)}
        }
    },
    avx512: {
        'required_alignment': {'bytes': 64},
        'vector_size': {'bytes': 64, 'bits': 512},
        'versions': {avx512: {'headers': ('<zmmintrin.h>',)}}
    }
}

to_family_arch = {v: fms for fms, info in family_archs.iteritems() for v in info['versions']}  # get family sse/avx/...
get_header_files = lambda arch: family_archs[to_family_arch[arch]]['versions'][arch]['headers']
get_switches = lambda arch: family_archs[to_family_arch[arch]]['versions'][arch]['switches']
get_required_alignment_in_bytes = lambda arch, fam=family_archs: family_archs[arch]['required_alignment']['bytes']

vector_types_postfix = dict(
    chain((('ps', ''), ('pd', 'd')), ((pf, 'i') for ct, pf in postfixes.iteritems() if pf.startswith('ep')))
)
for fam_arch, info in family_archs.iteritems():
    info['intrinsic'] = {'prefix': common_prefixes[fam_arch]}
    info['vector_type_prefix'] = '__m' + str(info['vector_size']['bits'])
    info['vector_types'] = {
        ct: (info['vector_type_prefix'] + vector_types_postfix[pf]) for ct, pf in postfixes.iteritems()
    }
    for v_name, v_info in info['versions'].iteritems():
        v_info['switches'] = {
            'gcc': ('-m' + v_name,),
            'clang': ('-m' + v_name,),
            'icc': (('-x' + v_name).replace(avx2, 'core-' + avx2),)
        }
supported_compilers = 'icc', 'gcc', 'clang'

intrinsic_info = lambda arch, family_archs=family_archs: family_archs[arch]['intrinsic']
__intrs_info = lambda arch, intrs_name, postfix, return_type, operand_types, extra=None: dict(
    chain((
        ('name', '{p}{o}{t}'.format(p=intrinsic_info(arch)['prefix'], o=intrs_name, t=postfix)),
        ('return_type', return_type),
        ('operand_type', operand_types),),
        (extra or {}).iteritems()
    ))


load_intrinsics = {
    sse: {
        'aligned': __intrs_info(sse, '_load_', 'ps', '__m128', ('float *',), extra={
            'alternative': __intrs_info(sse, '_stream_load_', 'si128', '__m128i', ('__m128i *',),
                                        extra={'arch': sse4_1})
        }),
        'unaligned': __intrs_info(sse, '_loadu_', 'si128', '__m128', ('float *',))
    },
    avx: {
        'aligned': __intrs_info(avx, '_load_', 'si256', '__m256i', ('const __m256i *',), extra={
            'alternative': __intrs_info(avx, '_stream_load_', 'si256', '__m256i', ('const __m256i *',),
                                        extra={'arch': avx2})
        }),
        'unaligned': __intrs_info(avx, '_lddqu_', 'si256', '__m256i', ('const __m256i *',))
    },
    avx512: {
        'aligned': __intrs_info(avx512, '_stream_load_', 'si512', '__m512i', ('const void *',)),
        'unaligned': __intrs_info(avx512, '_loadu_', 'si512', '__m512i', ('const void *',))
    }
}

store_intrinsics = {
    sse: {
        'aligned': __intrs_info(sse, '_stream_', 'ps', 'void', ('float *', '__m128',), extra={
            'alternative': __intrs_info(sse, '_stream_', 'si128', 'void', ('__m128i *', '__m128i'),
                                        extra={'arch': sse2})
        }),
        'unaligned': __intrs_info(sse, '_storeu_', 'ps', 'void', ('float *', '__m128'), extra={
            'alternative': __intrs_info(sse, '_storeu_', 'si128', 'void', ('__m128i *', '__m128i'),
                                        extra={'arch': sse2})
        })
    },
    avx: {
        'aligned': __intrs_info(avx, '_stream_', 'si256', 'void', ('__m256i *', '__m256i')),
        'unaligned': __intrs_info(avx, '_storeu_', 'si256', 'void', ('__m256i *', '__m256i'))
    },
    avx512: {
        'aligned': __intrs_info(avx512, '_stream_', 'si512 ', 'void', ('void *', '__m512i')),
        'unaligned': __intrs_info(avx512, '_storeu_', 'si512', 'void', ('void *', '__m512i'))
    }
}

set1_intrinsic_postfixes = dict((ct, (p + 'x' if p == 'epi64' else p)) for ct, p in postfixes.iteritems())
set1_intrinsics = {
    arch: dict(
        (ct, __intrs_info(arch, '_set1_',
                          p.replace('x', '') if arch == avx512 and p == 'epi64x' else p,
                          family_archs[arch]['vector_types'][ct], (ct,)))
        for ct, p in set1_intrinsic_postfixes.iteritems()
    ) for arch, pref in common_prefixes.iteritems()
}


_build_intrinsic_info = lambda nm, op_type, ret_type: {'name': nm, 'operand_type': op_type, 'return_type': ret_type}
_intrinsic_per_arch_ = lambda archs, supported_types, oper_name, number_of_operands=1: {
    ar: {c: _build_intrinsic_info(
        pref + oper_name + p,
        (number_of_operands * (family_archs[ar]['vector_types'][c],)),
        family_archs[ar]['vector_types'][c],
    ) for c, p in supported_types.iteritems()} for ar, pref in archs.iteritems()
}

sqrt_intrinsics = _intrinsic_per_arch_(common_prefixes, real_postfixes, '_sqrt_')
recp_intrinsics = _intrinsic_per_arch_(common_prefixes, {'float': real_postfixes['float']}, '_recp_')
recp_intrinsics[avx512]['float']['name'] = '_mm512_rcp28_ps'

add_intrinsics, sub_intrinsics, min_intrinsics, max_intrinsics = (
    _intrinsic_per_arch_(common_prefixes, postfixes, oper_name, 2) for oper_name in ('_add_', '_sub_', '_min_', '_max_')
)
mul_intrinsics, div_intrinsics = imap(
    _intrinsic_per_arch_, repeat(common_prefixes), repeat(real_postfixes), ('_mul_', '_div_'), repeat(2)
)
for arch, pref in common_prefixes.iteritems():
    mul_intrinsics[arch].update((ct,  _build_intrinsic_info(
        (pref + '_mullo_' + postfixes[ct]),
        2 * (family_archs[arch]['vector_types'][ct],),
        (family_archs[arch]['vector_types'][ct],)
    )) for ct in ('int', 'short'))

real_sizes = dict(imap(
    reversed,
    ifilter(lambda i: i[0] in real_postfixes, ((n, s['bytes'])for n, s in c_type_sizes.iteritems()))
))
int_as_real_of_same_size_postfixes = dict(
    (ct_name, postfixes[real_sizes[c_type_sizes[ct_name]['bytes']]])
    for ct_name in ifilter(lambda i: c_type_sizes[i]['bytes'] in real_sizes, postfixes.iterkeys())
)

and_intrinsics, or_intrinsics, xor_intrinsics, andnot_intrinsics = (
    _intrinsic_per_arch_(common_prefixes, int_as_real_of_same_size_postfixes, oper_name, 2)
    for oper_name in ('_and_', '_or_', '_xor_', '_andnot_')
)
bitwise_intrinsics = and_intrinsics, or_intrinsics, xor_intrinsics, andnot_intrinsics

shift_logical_left_by_immediate_intrinsics, shift_logical_right_by_immediate_intrinsics = imap(
    _intrinsic_per_arch_, repeat(common_prefixes), repeat(integral_postfixes), ('_slli_', '_srli_')
)
shift_intrinsics = shift_logical_left_by_immediate_intrinsics, shift_logical_right_by_immediate_intrinsics
for shift_instr in shift_intrinsics:
    for ar_name, ar_info in family_archs.iteritems():
        for c_type_name in integral_postfixes.iterkeys():
            shift_instr[ar_name][c_type_name]['operand_type'] = (
                shift_instr[ar_name][c_type_name]['operand_type'][0], 'int'
            )
bitwise_intrinsics += shift_intrinsics


def subprocess_call_consume_os_error(*args, **kwargs):
    try:
        return subprocess.call(*args, **kwargs)
    except OSError:
        return -1


def get_all_intrinsics(arch_version):
    # get all intrinsic(s) from a specific arch version (sse, sse2, sse3, ..., avx2, ...)
    header_files = get_header_files(arch_version)
    with tempfile.NamedTemporaryFile(bufsize=0) as output_file:
        with tempfile.NamedTemporaryFile(bufsize=0) as temp_file:
            temp_file.write(os.linesep.join(imap('#include {0}'.format, header_files)) + os.linesep)
            temp_file.flush()
            ret_code = next(ifilterfalse(  # go through each possible compiler
                None,  # check the return code, only use the output from a call that returned zero.
                (subprocess_call_consume_os_error(
                    tuple(
                        chain((c, '-xc', '-E', temp_file.name, '-o', output_file.name), get_switches(arch_version)[c])
                    ),
                    stderr=tempfile.NamedTemporaryFile(),
                    stdout=tempfile.NamedTemporaryFile()
                ) for c in supported_compilers)), 1)  # if they all failed just return 1 ...
        content = (not ret_code and open(output_file.name, 'r').read()) or ''
    if not content:
        pass  # all compilers failed TODO: check online if we have internet access ...
    content += ''.join(
        open(f, 'r').read() for f in ifilter(
            os.path.isfile,
            (  # some of the intrinsics are #define so they will be omitted when pre-processed look for the actual file
                g.group()[1:-1]  # path in the pre-processed output and go through the actual source code ...
                for header_file in ifilter(  # header file path of existing file paths in preprocessed output ...
                    lambda hd_file, content=content: os.path.sep + hd_file in content,
                    imap(lambda h: h.replace('<', '').replace('>', ''), header_files)  # file name
                ) for g in re.compile('\".+\{0}{1}\"'.format(os.path.sep, header_file)).finditer(content)  # file path
            )
        )
    )
    prefix = family_archs[to_family_arch[arch_version]]['intrinsic']['prefix']
    postfix = chain(
        postfixes.itervalues(),
        (p.replace('epi', 'epu') for p in postfixes.itervalues()),  # epu for unsigned integrals ...
        ('epi64x',),
        ('si' + str(byte_bit_size * family_archs[to_family_arch[arch_version]]['vector_size']['bytes']),)
    )
    eng = re.compile('{0}[a-zA-Z_0-9]+_({1}){{1,1}}'.format(prefix, '|'.join(
        sorted(set(postfix), key=len, reverse=True)
    )))
    return (g.group() for g in eng.finditer(content or ''))


get_max_arch = lambda sorted_archs=sorted(
    family_archs.iterkeys(), key=lambda k: family_archs[k]['vector_size']['bytes'], reverse=True
): next(ifilter(does_cpu_support, sorted_archs), None)

largest_native_arch = get_max_arch()
all_natively_supported_instruction_sets = set(ifilter(does_cpu_support, instruction_sets))
all_natively_supported_family_archs = set(filter(
    all_natively_supported_instruction_sets.__contains__, family_archs.iterkeys()
))
all_natively_supported_intrinsics = set(chain.from_iterable(
    imap(get_all_intrinsics, all_natively_supported_instruction_sets)
))
all_natively_supported_headers = set(
    chain.from_iterable(imap(get_header_files, all_natively_supported_instruction_sets))
)
all_natively_supported_switches = defaultdict(list)
all_natively_supported_compilers = tuple(ifilterfalse(lambda s: subprocess_call_consume_os_error(
    (s, '--version'), stdout=tempfile.NamedTemporaryFile(), stderr=tempfile.NamedTemporaryFile()
), supported_compilers))
for switches in imap(get_switches, all_natively_supported_instruction_sets):
    for cc, cc_switches in switches.iteritems():
        all_natively_supported_switches[cc].extend(cc_switches)
all_natively_supported_switches['icc'] = '-xHost',


func_signature = lambda func_name, return_type, operands: \
    'static {inline} {return_type} {func_name}({operands})'.format(
        func_name=func_name,
        return_type=return_type,
        inline='__inline__ __attribute__((__always_inline__))',
        operands=','.join(operands)
    )


# define new intrinsic(s) using other intrinsic(s) eg: _square_(val) => mul_instr(val, val)
user_defined_intrinsics_code = \
    os.linesep.join(
        imap(
            os.linesep.join,
            (
                ('#define {pr}_not_si{s}(val) ({pr}_xor_si{s}(val, {pr}_set1_epi32(-1)))'.format(
                    pr=arch_info['intrinsic']['prefix'], s=arch_info['vector_size']['bits'])
                 for arch, arch_info in family_archs.iteritems()),
                #define {pr}_square_{pf}(val) ({mul_instr}(val, val)) {return val, val}
            )
        )
    )


not_instrs = {
    arch: {
        ct: {
            'name': '{0}_not_si{1}'.format(arch_info['intrinsic']['prefix'], arch_info['vector_size']['bits']),
            'operand_type': ('',),
            'return_type': '',
            'arch': arch + '2' if arch != avx512 else '',
        }
        for ct, _ in postfixes.iteritems()
    } for arch, arch_info in family_archs.iteritems()
}


user_defined_instrs = not_instrs,  # square_instrs
all_natively_supported_intrinsics |= set(
    instr_info['name'] for instr_info in ifilter(  # add the user defined intrinsic(s) if the architecture is available
        lambda instrs: instrs['arch'] in all_natively_supported_instruction_sets,
        chain.from_iterable(info.itervalues() for user_def_i in user_defined_instrs for info in user_def_i.itervalues())
    )
)

all_intrinsics = {
    # unary move operations ...
    'load': load_intrinsics,
    'store': store_intrinsics,
    'set1': set1_intrinsics,

    # unary arithmetic operations ...
    'sqrt': sqrt_intrinsics,
    'recp': recp_intrinsics,

    'not': not_instrs,
    # 'square': square_instrs,

    # binary arithmetic operations ...
    'add': add_intrinsics,
    'sub': sub_intrinsics,
    'mul': mul_intrinsics,
    'div': div_intrinsics,

    # binary bitwise operations ...
    'and': and_intrinsics,
    'or': or_intrinsics,
    'xor': xor_intrinsics,
    'andnot': andnot_intrinsics,

    'min': min_intrinsics,
    'max': max_intrinsics,

    'shift_left_logical_immediate': shift_logical_left_by_immediate_intrinsics,
    'shift_right_logical_immediate': shift_logical_right_by_immediate_intrinsics

}

for c_type_name, c_type_pf in real_postfixes.iteritems():
    # TODO: Update operand types!
    _info = _intrinsic_per_arch_(common_prefixes, {'int': postfixes['int']}, '_cvt{0}_'.format(c_type_pf))

    all_intrinsics['convert_{0}_to_int'.format(c_type_name.replace(' ', '_'))] = _intrinsic_per_arch_(
        common_prefixes, {'int': postfixes['int']}, '_cvt{0}_'.format(c_type_pf)
    )
    all_intrinsics['convert_int_to_{0}'.format(c_type_name.replace(' ', '_'))] = _intrinsic_per_arch_(
        common_prefixes,
        {c_type_name: postfixes[c_type_name]},
        '_cvt{0}_'.format(postfixes['int'])
    )


operations_to_intrinsics = {  # use the same names as numpy so we can fall back on it if not supported.
    'add': 'add', 'subtract': 'sub', 'multiply': 'mul', 'divide': 'div',  # binary arithmetic ...
    'bitwise_and': 'and', 'bitwise_or': 'or', 'bitwise_xor': 'xor', 'bitwise_andnot': 'andnot',  # binary bitwise
    'sqrt': 'sqrt', 'reciprocal': 'recp',  # unary arithmetic
    'invert': 'not',   # unary bitwise ...
}


for arch, intr_info in ifilter(lambda item: item[0] in all_natively_supported_family_archs, mul_intrinsics.iteritems()):
    for ct, instr_info in intr_info.iteritems():
        instrs_operation_name = 'square'
        intrinsic_name = '{pr}_square_{pf}'.format(pr=family_archs[arch]['intrinsic']['prefix'], pf=postfixes[ct])
        code = '{func} {{ return {mul_instr}(val, val); }}'.format(
            func=func_signature(
                intrinsic_name,
                family_archs[arch]['vector_types'][ct],
                ('{0} val'.format(family_archs[arch]['vector_types'][ct]),)
            ),
            mul_instr=instr_info['name']
        )
        required_intrinsics = {'mul': instr_info['name']}
        if all(imap(all_natively_supported_intrinsics.__contains__, required_intrinsics.itervalues())):
            user_defined_intrinsics_code += os.linesep + code
            all_natively_supported_intrinsics |= {intrinsic_name}
            all_intrinsics.setdefault(instrs_operation_name, {}).setdefault(arch, {})[ct] = \
                _build_intrinsic_info(
                    intrinsic_name,
                    (family_archs[arch]['vector_types'][ct],),
                    (family_archs[arch]['vector_types'][ct],)
                )
            operations_to_intrinsics[instrs_operation_name] = instrs_operation_name


ieee_real_info = {
    'float': {'exponent': {'bias': 127, 'offset': {'bits': 23}}},
    'double': {'exponent': {'bias': 1023, 'offset': {'bits': 52}}}
}
for name, info in ieee_real_info.iteritems():
    info['postfix'] = real_postfixes[name]


def sse_poly_eval_code(coefficients, var_name, mul_intrs, add_intrs, set1_intrs):
    return '{set1_intrs}({c:.26f})'.format(set1_intrs=set1_intrs, c=coefficients[0]) if len(coefficients) == 1 \
        else '{add_intrs}({mul_intrs}({left_operand}, {var_name}), {set1_intrs}({c0:.26f}))'.format(
            add_intrs=add_intrs, mul_intrs=mul_intrs, set1_intrs=set1_intrs, var_name=var_name, c0=coefficients[-1],
            left_operand=sse_poly_eval_code(coefficients[:-1], var_name, mul_intrs, add_intrs, set1_intrs)
        )


exp2_polynomials_descending = (  # remez(numpy.exp2, (-.25, .25), 8, 2**-52)
    1.320645534568377645848625e-06,   1.526704106755039661242127e-05,
    1.540354777391378122280985e-04,   1.333355144518069430176199e-03,
    9.618129100738954939675551e-03,   5.550410867644831741651856e-02,
    2.402265069591746904364982e-01,   6.931471805598905522316500e-01,
    1.000000000000000000000000e+00
),

exp2_real_template = """
{func_sig}  {{
    // 2**floor(y) is calculated using shifts, since floor(y) is an integral
    // the second term is calculated using a polynomial interpolation of 2**x on the interval -0.25 to 0.25
    // since: -0.25 < 0.25*(y - floor(y)) < 0.25
    // the coefficients where located using the remez algorithm ...

    {integral_vt} integral_part = {convert_to_integral}(x); // convert to integer ...

    // get the fractional part divided into quarters,
    // its ok as long as we double sqrt the integral part and double square the final product
    {float_vt} fractional_part = {mul_floats}({sub_floats}(x, {convert_to_floats}(integral_part)), {set1_float}(0.25));

    {float_vt} result = {mul_floats}( // exponent_part
        {sqrt_float}({sqrt_float}( // take the square root so we can cut the interval in half ...
            // interpret_as_floats( //_mm_castsi128_pd Set the exponent part of each float value .. (1 << integral_part)
            {shift_left}(      // Since the exponent is already in base two we simply need to add the bias and shift
                {add_integrals}({integral_part}, {set1_integral}({ieee_exponent_bias})),
                {ieee_exponent_offset}  // Move the exponent to its proper location ...
            )
            // )
        )),
        {poly_eval}
    );
    result = {mul_floats}(result, result);  // double square the result to get final result ...
    return {mul_floats}(result, result);
}}
"""
for poly_coefs, c_type_pf, arch in product(
        exp2_polynomials_descending,
        real_postfixes.iteritems(),
        imap(lambda n: (n, family_archs[n]), all_natively_supported_family_archs)
):
    c_type_name, c_type_pf = c_type_pf
    arch_family_name, arch_family_info = arch
    arch_prefix = arch_family_info['intrinsic']['prefix']
    integral_c_type_name = next(ifilter(
        lambda n, ct=c_type_name: c_type_sizes[n] == c_type_sizes[c_type_name], integral_postfixes.iterkeys()
    ))
    shuffle_instr = '{prefix}_shuffle_epi32'.format(prefix=arch_prefix)
    intrinsic_name = '{arch_prefix}_exp2_poly_{degree}_{postfix}'.format(
        arch_prefix=arch_prefix, degree=len(poly_coefs) - 1, postfix=c_type_pf
    )
    integral_part = 'integral_part'
    if c_type_sizes[c_type_name] != c_type_sizes['int']:
        # unfortunately there is no way to convert a pack double to epi64 so we need to shuffle the words around ...
        vector_byte_size = arch_family_info['vector_size']['bytes']
        number_of_ints_per_vector = vector_byte_size/c_type_sizes[c_type_name]['bytes']
        integral_part = '{shuffle_instr}({n}, _MM{s}_SHUFFLE({shuffle}))'.format(
            shuffle_instr=shuffle_instr, prefix=arch_prefix, n=integral_part,
            s='' if vector_byte_size == 16 else (vector_byte_size*byte_bit_size),
            shuffle=','.join(imap(str, reversed(tuple(chain.from_iterable(  # swap all the odd words
                izip(xrange(0, number_of_ints_per_vector), xrange(number_of_ints_per_vector, vector_byte_size))
            )))))
        )

    required_intrinsics = {
        'mul_floats': all_intrinsics['mul'][arch_family_name][c_type_name]['name'],
        'set1_float': all_intrinsics['set1'][arch_family_name][c_type_name]['name'],
        'add_floats': all_intrinsics['add'][arch_family_name][c_type_name]['name'],
        'sub_floats': all_intrinsics['sub'][arch_family_name][c_type_name]['name'],
        'add_integrals': all_intrinsics['add'][arch_family_name][integral_c_type_name]['name'],
        'set1_integral': all_intrinsics['set1'][arch_family_name][integral_c_type_name]['name'],
        'shift_left': all_intrinsics['shift_left_logical_immediate'][arch_family_name][integral_c_type_name]['name'],
        'sqrt_float': all_intrinsics['sqrt'][arch_family_name][c_type_name]['name'],
        'floats_min': all_intrinsics['min'][arch_family_name][c_type_name]['name'],
        'floats_max': all_intrinsics['max'][arch_family_name][c_type_name]['name'],
        'convert_to_floats': all_intrinsics['convert_int_to_{0}'.format(c_type_name)]
        [arch_family_name][c_type_name]['name'],
        'convert_to_integral': all_intrinsics['convert_{0}_to_int'.format(c_type_name)]
        [arch_family_name]['int']['name']
    }

    if shuffle_instr in integral_part:
        required_intrinsics['__shuffle_instr__'] = shuffle_instr

    template_vars = dict(
        chain(
            (
                ('func_sig', func_signature(
                    intrinsic_name,
                    arch_family_info['vector_types'][c_type_name],
                    (' '.join((arch_family_info['vector_types'][c_type_name], 'x')),)
                )),
                ('instr_name', intrinsic_name),
                ('integral_part', integral_part),
                ('float_vt', arch_family_info['vector_types'][c_type_name]),
                ('integral_vt', arch_family_info['vector_types']['int']),
                ('exponent_upper_bound', ieee_real_info[c_type_name]['exponent']['bias'] + 1),
                ('exponent_lower_bound', -ieee_real_info[c_type_name]['exponent']['bias']),
                ('ieee_exponent_offset', ieee_real_info[c_type_name]['exponent']['offset']['bits']),
                ('ieee_exponent_bias', ieee_real_info[c_type_name]['exponent']['bias']),
                ('poly_eval', sse_poly_eval_code(
                    poly_coefs,
                    'fractional_part',
                    required_intrinsics['mul_floats'],
                    required_intrinsics['add_floats'],
                    required_intrinsics['set1_float'],
                )),
            ),
            imap(
                lambda n, required_intrinsics=required_intrinsics: (n, required_intrinsics[n]),
                required_intrinsics.iterkeys()
            )
        )
    )
    if all(imap(all_natively_supported_intrinsics.__contains__, required_intrinsics.itervalues())):
        user_defined_intrinsics_code += exp2_real_template.format(**template_vars)
        instrs_operation_name = 'exp2_poly_{0}'.format(len(poly_coefs) - 1)
        all_natively_supported_intrinsics |= {intrinsic_name}
        all_intrinsics.setdefault(instrs_operation_name, {}).setdefault(arch_family_name, {})[c_type_name] = \
            _build_intrinsic_info(
                intrinsic_name,
                (arch_family_info['vector_types'][c_type_name],),
                (arch_family_info['vector_types'][c_type_name],)
            )
        operations_to_intrinsics[instrs_operation_name] = instrs_operation_name


exp_real_template = """
#define {instr_name}(x) {exp2}(\
    {mul_floats}(x, {set1_float}(1.442695040888963387004650940071)\
    )  \
)// multiply by log2(e)
"""
for exp2_func, c_type_pf, arch in product(
        ifilter(lambda n: n.startswith('exp2_poly'), all_intrinsics.iterkeys()),
        real_postfixes.iteritems(),
        imap(lambda n: (n, family_archs[n]), all_natively_supported_family_archs)
):
    c_type_name = c_type_pf[0]
    arch_family_name, arch_family_info = arch
    arch_prefix = arch_family_info['intrinsic']['prefix']

    required_intrinsics = {
        'mul_floats': all_intrinsics['mul'][arch_family_name][c_type_name]['name'],
        'set1_float': all_intrinsics['set1'][arch_family_name][c_type_name]['name'],
        'exp2': all_intrinsics[exp2_func][arch_family_name][c_type_name]['name']
    }
    intrinsic_name = all_intrinsics[exp2_func][arch_family_name][c_type_name]['name'].replace('exp2', 'exp')
    template_vars = dict(chain((('instr_name', intrinsic_name),), required_intrinsics.iteritems()))

    if all(imap(all_natively_supported_intrinsics.__contains__, required_intrinsics.itervalues())):
        user_defined_intrinsics_code += exp_real_template.format(**template_vars)
        instrs_operation_name = exp2_func.replace('exp2', 'exp')
        all_natively_supported_intrinsics |= {intrinsic_name}
        all_intrinsics.setdefault(instrs_operation_name, {}).setdefault(arch_family_name, {})[c_type_name] = \
            _build_intrinsic_info(
                intrinsic_name,
                (arch_family_info['vector_types'][c_type_name],),
                (arch_family_info['vector_types'][c_type_name],)
            )
        operations_to_intrinsics[instrs_operation_name] = instrs_operation_name


log2_polynomials_descending = (  # remez(lambda v: numpy.log2(v)/(v - 1.0), interval, 8, error=2**-51)
    7.395401787399079329698992e-03,  -1.011082226447333337615575e-01,
    6.125183829165274929096086e-01,  -2.162219075865688733273373e+00,
    4.919650517166084036091434e+00,  -7.540432746276700903820256e+00,
    7.934690466015142717992603e+00,  -5.863444848987134250251074e+00,
    3.635645117619682231691058e+00
),
log2_real_template = """
{func_signature} {{
    // get fractional part, value between 1, 2
    {float_vt} mantissa = {or_float}(  // get 1.0 + sum(b2**-i, i=1..len(mantissa))
        {and_float}(x, {set1_integral}({mantissa_mask})), // get mantissa
        {set1_float}(1)
    );
    {integral_vt} exponent = {sub_integrals}(
        {shift_right}(
            {and_float}(x, {set1_integral}({ieee_exponent_mask})),  // get exponent and remove all other bits
            {ieee_exponent_offset}
        ),
        {set1_integral}({ieee_float_bias})
    );

    return {add_floats}(
        {convert_to_floats}({exponent_part}), // get exponent which is already in log2 format ...
        {mul_floats}({polyval}, {sub_floats}(mantissa, {set1_float}(1)))
    );
}}
"""
for poly_coefs, c_type_pf, arch in product(
    log2_polynomials_descending,
    real_postfixes.iteritems(),
    imap(lambda n: (n, family_archs[n]), all_natively_supported_family_archs)
):
    c_type_name, ct_postfix = c_type_pf
    arch_family_name, arch_family_info = arch
    float_vt = arch_family_info['vector_types'][c_type_name]
    arch_prefix = arch_family_info['intrinsic']['prefix']
    integral_c_type_name = next(ifilter(
        lambda n, ct=c_type_name: c_type_sizes[n] == c_type_sizes[c_type_name], integral_postfixes.iterkeys()
    ))

    shuffle_instr = '{prefix}_shuffle_epi32'.format(prefix=arch_prefix)
    exponent_part = 'exponent'
    if c_type_sizes[c_type_name]['bytes'] != c_type_sizes['int']['bytes']:
        # unfortunately there is no way to convert epi64 to pd so we need to shuffle the words around ...
        vector_byte_size = arch_family_info['vector_size']['bytes']
        number_of_ints_per_vector = vector_byte_size/c_type_sizes[c_type_name]['bytes']
        exponent_part = '{shuffle_instr}({n}, _MM{s}_SHUFFLE({shuffle}))'.format(
            shuffle_instr=shuffle_instr, prefix=arch_prefix, n=exponent_part,
            s='' if vector_byte_size == 16 else (vector_byte_size*byte_bit_size),
            shuffle=','.join(imap(str, reversed(tuple(chain.from_iterable(  # swap all the odd words
                izip(xrange(0, number_of_ints_per_vector), xrange(number_of_ints_per_vector, vector_byte_size))
            )))))
        )

    required_intrinsics = {
        'or_float': all_intrinsics['or'][arch_family_name][c_type_name]['name'],
        'and_float': all_intrinsics['and'][arch_family_name][c_type_name]['name'],
        'set1_float': all_intrinsics['set1'][arch_family_name][c_type_name]['name'],
        'add_floats': all_intrinsics['add'][arch_family_name][c_type_name]['name'],
        'sub_floats': all_intrinsics['sub'][arch_family_name][c_type_name]['name'],
        'mul_floats': all_intrinsics['mul'][arch_family_name][c_type_name]['name'],
        'sub_integrals': all_intrinsics['sub'][arch_family_name][integral_c_type_name]['name'],
        'set1_integral': all_intrinsics['set1'][arch_family_name][integral_c_type_name]['name'],
        'shift_right': all_intrinsics['shift_right_logical_immediate'][arch_family_name][integral_c_type_name]['name'],
        'convert_to_floats':
        all_intrinsics['convert_int_to_{0}'.format(c_type_name)][arch_family_name][c_type_name]['name'],
    }
    intrinsic_name = '{arch_prefix}_log2_poly_{degree}_{postfix}'.format(
        arch_prefix=arch_prefix, degree=len(poly_coefs) - 1, postfix=ct_postfix
    )
    exponent_bit_offset = ieee_real_info[c_type_name]['exponent']['offset']['bits']
    exponent_bias = ieee_real_info[c_type_name]['exponent']['bias']
    template_vars = dict(
        chain(
            (
                ('exponent_part', exponent_part),
                ('integral_vt', arch_family_info['vector_types'][integral_c_type_name]),
                ('float_vt', float_vt),
                ('func_signature', func_signature(intrinsic_name, float_vt, (' '.join((float_vt, 'x')),))),
                ('instr_name', intrinsic_name),
                ('ieee_float_bias', exponent_bias),
                ('ieee_exponent_offset', exponent_bit_offset),
                ('mantissa_mask',
                    '{0}{1}'.format(
                        hex((1 << exponent_bit_offset) - 1), '' if exponent_bit_offset < c_type_sizes['int']['bits'] else 'LL'
                    )),
                ('ieee_exponent_mask',
                    '{0}{1}'.format(
                        hex((exponent_bias | (exponent_bias + 1)) << exponent_bit_offset),
                        '' if (exponent_bit_offset + numpy.log2((exponent_bias + 1) * 2)) < c_type_sizes['int']['bits']
                        else 'LL'
                    )),
                ('polyval', sse_poly_eval_code(
                    poly_coefs,
                    'mantissa',
                    all_intrinsics['mul'][arch_family_name][c_type_name]['name'],
                    all_intrinsics['add'][arch_family_name][c_type_name]['name'],
                    all_intrinsics['set1'][arch_family_name][c_type_name]['name'],
                ))
            ),
            required_intrinsics.iteritems()
        )
    )

    if all(imap(all_natively_supported_intrinsics.__contains__, required_intrinsics.itervalues())):
        user_defined_intrinsics_code += log2_real_template.format(**template_vars)
        instrs_operation_name = 'log2_poly_{0}'.format(len(poly_coefs) - 1)
        all_natively_supported_intrinsics |= {intrinsic_name}
        all_intrinsics.setdefault(instrs_operation_name, {}).setdefault(arch_family_name, {})[c_type_name] = \
            _build_intrinsic_info(
                intrinsic_name,
                (arch_family_info['vector_types'][c_type_name],),
                (arch_family_info['vector_types'][c_type_name],)
            )
        operations_to_intrinsics[instrs_operation_name] = instrs_operation_name

log_real_template = """
// 0.693147180559945286226764 == 1/log2(e), log(x) == log2(x)/log2(e)
#define {instr_name}(x) {mul_floats}(\
        {log2_floats}(x),\
        {set1_float}(0.693147180559945286226764)\
    )
"""
for poly_coefs, c_type_pf, arch in product(
    log2_polynomials_descending,
    real_postfixes.iteritems(),
    imap(lambda n: (n, family_archs[n]), all_natively_supported_family_archs)
):
    arch_family_name, arch_family_info = arch
    c_type_name, c_type_pf = c_type_pf
    intrinsic_name = '{arch}_log_poly_{degree}_{pf}'.format(
        arch=common_prefixes[arch_family_name], degree=len(poly_coefs) - 1, pf=c_type_pf
    )
    required_intrinsics = {
        'mul_floats': all_intrinsics['mul'][arch_family_name][c_type_name]['name'],
        'set1_float': all_intrinsics['set1'][arch_family_name][c_type_name]['name'],
        'log2_floats': intrinsic_name.replace('log', 'log2')
    }
    template_vars = dict(chain((('instr_name', intrinsic_name),), required_intrinsics.iteritems()))

    if all(imap(all_natively_supported_intrinsics.__contains__, required_intrinsics.itervalues())):
        user_defined_intrinsics_code += log_real_template.format(**template_vars)
        instrs_operation_name = 'log_poly_{0}'.format(len(poly_coefs) - 1)
        all_natively_supported_intrinsics |= {intrinsic_name}
        all_intrinsics.setdefault(instrs_operation_name, {}).setdefault(arch_family_name, {})[c_type_name] = \
            _build_intrinsic_info(
                intrinsic_name,
                (arch_family_info['vector_types'][c_type_name],),
                (arch_family_info['vector_types'][c_type_name],)
            )
        operations_to_intrinsics[instrs_operation_name] = instrs_operation_name