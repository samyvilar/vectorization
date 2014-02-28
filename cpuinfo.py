__author__ = 'samyvilar'

import platform
import subprocess
import string
from itertools import imap, chain, repeat, ifilter, izip, count
from compile import compile_and_load
from ctypes import c_int

import logging

logger = logging.basicConfig(format='%(asctime)-15s %(module)s: %(message)s')

apply_funcs = lambda value, funcs=(): imap(apply, funcs, repeat((value,)))

convert_delimited_string_to_set = lambda s, d=' ', funcs=(string.lower, string.upper): \
    set(chain.from_iterable(
        apply_funcs(v, funcs) for v in ifilter(None, imap(string.strip, s.split(d)))))

replaces_sub_strs = lambda s, p=(), n='': reduce(lambda s, p, n=n: s.replace(p, n), p, s)
split_by_any = lambda s, d=('',): s.split(next(ifilter(lambda d: d in s, d), ''))


def warn(msg):
    logging.warning(msg, {'module': 'cpuinfo'})


common_property_name = \
    'vendor', \
    'model', \
    'microcode', \
    'cpu_frequency', \
    'features', \
    'l3_cache_size', \
    'physical_cpu_count', \
    'available_cpu_count',      \
    'physical_address_size', \
    'virtual_address_size'


def get_darwin_info():
    set_properties = 'features', 'extfeatures'
    prefixes = 'machdep.cpu.', 'hw.'
    delimiters = '=', ':'
    mapping_to_common_properties = {
        'vendor': 'vendor',
        'model': 'brand_string',
        'microcode': 'microcode_version',
        'cpu_frequency': 'cpufrequency',
        'features': 'features',
        'l3_cache_size': 'l3cachesize',
        'available_cpu_count': 'availcpu',
        'physical_address_size': 'address_bits.physical',
        'virtual_address_size': 'address_bits.virtual'

    }
    info = {
        replaces_sub_strs(full_name, prefixes).strip(): value.replace('  ', '').strip()
        for full_name, value in (
            split_by_any(line, delimiters) for line in
            (subprocess.check_output(('sysctl', '-a', 'machdep.cpu')) +
             subprocess.check_output(('sysctl', '-a', 'hw'))).splitlines()
        )
    }
    info.update((p, convert_delimited_string_to_set(info[p])) for p in set_properties)
    info['features'] = info['features'] | info['extfeatures']
    info.update(
        chain(
            info.iteritems(),
            izip(mapping_to_common_properties, imap(info.get, mapping_to_common_properties.itervalues()))
        )
    )

    return info


def get_linux_info():
    set_properties = 'flags',
    mapping_to_common_properties = {
        'vendor': 'vendor_id',
        'model': 'model name',
        'microcode': 'microcode',
        'features': 'flags',
    }
    processors = tuple(
        imap(string.split, ifilter(None, subprocess.check_output(('cat', '/proc/cpuinfo')).split('\n\n')), repeat('\n'))
    )

    info = {
        str(cpu_numb): {
            full_name.strip(): value.replace('  ', '').strip()
            for full_name, value in imap(string.split, cpu, repeat(':'))
        } for cpu_numb, cpu in enumerate(processors)
    }
    info['cpu_frequency'] = str(float(info['cpu MHz']) * 10**6)
    info['available_cpu_count'] = len(processors)
    value, factor = info['cache size'].split(' ')
    factors = {'': 1, 'B': 1, 'KB': 10**3, 'MB': 10**6, 'GB': 10**9}
    info['l3_cache_size'] = str(int(value) * factors[factor])
    phys_addr_size, virt_addr_size = imap(string.strip, info['address sizes'].split(','))
    info['physical_address_size'] = phys_addr_size.replace(' bits physical', '')
    info['virtual_address_size'] = virt_addr_size.replace(' bits virtual', '')
    info['cpu_info'] = processors

    info.update((p, convert_delimited_string_to_set(
        info[p] + ''.join(imap(lambda feat: ' ' + feat.replace('_', '.') + ' ', info[p].split(' ')))
    )) for p in set_properties)

    info.update(chain(
        info.iteritems(),
        izip(mapping_to_common_properties, imap(info.get, mapping_to_common_properties.itervalues()))
    ))

    return info


def get_windows_info():
    import _winreg

    def get_info(parent_key):
        def _get_all_stats(reg_entry):
            index = count()
            try:
                while True:
                    yield _winreg.EnumValue(reg_entry, next(index))
            except EnvironmentError as _:
                raise StopIteration
        cpu_ids = count()
        try:
            while True:
                cpu_id = next(cpu_ids)
                yield cpu_id, dict(
                    (entry, {'name': entry, 'value': value, 'type': _type})
                    for entry, value, _type in _get_all_stats(
                        _winreg.OpenKey(parent_key, _winreg.EnumKey(parent_key, cpu_id))
                    )
                )
        except EnvironmentError as _:
            raise StopIteration

    info = dict(get_info(r"HARDWARE\DESCRIPTION\System\CentralProcessor"))
    feature_names = 'MMX', 'SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE4_1', 'SSE4_2', 'AVX', 'FMA3', 'x64', 'SSE4a', \
                    'FMA4', 'XOP', 'AVX2'
    feature_ids = dict(izip(feature_names, count()))

    code = \
        """
        #define _cpuid_(cpuinfo, info_type) __asm__ __volatile__ ("cpuid": "=a" (cpuinfo[0]), "=b" (cpuinfo[1]), "=c" (cpuinfo[2]), "=d" (cpuinfo[3]) : "a" (info_type))

        void check_cpu_support(int features[24]) {{
            int info[4];
            _cpuid_(info, 0);
            int nIds = info[0];

            _cpuid_(info, 0x80000000);
            int nExIds = info[0];

            //  Detect Instruction Set
            if (nIds >= 1) {{
                _cpuid_(info, 0x00000001);
                features[{MMX}]   = (info[3] & ((int)1 << 23)) != 0;
                features[{SSE}]   = (info[3] & ((int)1 << 25)) != 0;
                features[{SSE2}]  = (info[3] & ((int)1 << 26)) != 0;
                features[{SSE3}]  = (info[2] & ((int)1 <<  0)) != 0;

                features[{SSSE3}] = (info[2] & ((int)1 <<  9)) != 0;
                features[{SSE4_1}] = (info[2] & ((int)1 << 19)) != 0;
                features[{SSE4_2}] = (info[2] & ((int)1 << 20)) != 0;

                features[{AVX}]   = (info[2] & ((int)1 << 28)) != 0;
                features[{FMA3}]  = (info[2] & ((int)1 << 12)) != 0;
            }}

            if (nExIds >= 0x80000001) {{
                _cpuid_(info, 0x80000001);
                features[{x64}]   = (info[3] & ((int)1 << 29)) != 0;
                features[{SSE4a}] = (info[2] & ((int)1 <<  6)) != 0;
                features[{FMA4}]  = (info[2] & ((int)1 << 16)) != 0;
                features[{XOP}]   = (info[2] & ((int)1 << 11)) != 0;
            }}
            _cpuid_(info, 0x00000007);
            features[{AVX2}]  = (info[2] & ((int)1 << 5)) != 0;
        }}
        """.format(**feature_ids)
    # noinspection PyCallingNonCallable
    features_c_param = (c_int * len(feature_names))(0)
    try:
        c = compile_and_load(code)
        c.check_cpu_support(features_c_param)
    except ValueError as er:
        warn('Failed to compile cpuid c code got: %s' % str(er))
    info['features'] = set(
        chain.from_iterable(
            imap(
                apply_funcs,
                chain.from_iterable(
                    ((feat_name, feat_name.replace('_', '.'))
                     for feat_name, feat_id in feature_ids.iteritems() if features_c_param[feat_id]),
                ),
                repeat(((string.lower, string.upper)))
            )
        )
    )

    return info


impls = {'Darwin': get_darwin_info, 'Linux': get_linux_info, 'Windows': get_windows_info}


def get_cpu_info(impls=impls):
    return impls[platform.system()]()


_info = get_cpu_info()


def does_cpu_support(feat_name, info=_info):
    return feat_name in info['features']


def get_all_cpu_features(info=_info):
    return info['features']