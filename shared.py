__author__ = 'samyvilar'

from multiprocessing import sharedctypes, cpu_count
from itertools import imap, chain
from ctypes_info import numpy_type_to_ctypes
import numpy


# SSE requires 16 bytes alignment, AVX requires 32, AVX-512 requires 64 ...
default_alignment = 64
default_segment_count = cpu_count()


def zeros(shape_or_count, dtype='float64', alignment=default_alignment, segment_count=default_segment_count):
    # At a minimum it allocates 160 bytes of shared memory
    count = numpy.product(shape_or_count)
    item_size = numpy.dtype(dtype).itemsize
    # Add alignment bytes for better SSE performance ...
    size_in_bytes = ((item_size * count) + (segment_count * alignment))  # add a couple bytes for alignment ...
    size_in_bytes -= item_size * (size_in_bytes/item_size % -segment_count)  # add a couple bytes for segmentation ...
    size_in_bytes -= item_size * ((size_in_bytes/item_size/segment_count) % -(alignment/item_size))
    assert not size_in_bytes % item_size  # make sure evenly create each item
    # assert not (size_in_bytes/item_size) % segment_count  # we can evenly divide by segments, IMPRACTICAL! ...
    raw_array = sharedctypes.RawArray('b', size_in_bytes)
    # noinspection PyNoneFunctionAssignment
    _buf = numpy.frombuffer(raw_array, dtype='b')
    start_index = -_buf.ctypes.data % alignment
    # get numpy array aligned ...
    return _buf[start_index:start_index + count*item_size].view(dtype).reshape(shape_or_count)


# numpy.array_split
def segment_array_into_ctypes(
        a, segment_count=default_segment_count, alignment=default_alignment, allotted_item_count=1
):
    """
        Segment array a into segment_count sub arrays, where each sub array address has
        an evenly divisible number items with respect to their item size and alignment ...
        @allotted_item_count: the number of 'extra' items that were allocated
        but not part of the vector, used for alignment, defaults to 1
    """
    flatten_a = a.ravel()                               # flatten array
    assert flatten_a.ctypes.data == a.ctypes.data       # make sure that flattening array doesn't a generate copy ...
    assert not flatten_a.ctypes.data % alignment        # make sure that the starting address is properly aligned ...

    number_of_items_per_alignment, rem = divmod(alignment, flatten_a.itemsize)
    assert not rem                                      # make sure that the item size is evenly divisible on alignment

    alignment_remainder = abs(flatten_a.size % -number_of_items_per_alignment)
    # make sure we can create at least one properly aligned array ...
    assert (flatten_a.size + alignment_remainder + allotted_item_count) >= number_of_items_per_alignment

    lengths, remainder = divmod(flatten_a.size, segment_count)
    counts = numpy.asarray([lengths] * segment_count)
    counts[-1] += remainder

    for index, count in enumerate(counts[:-1]):
        # iterate over sizes make sure each is evenly divisible over number_of_items_per_alignment ...
        # either shrink or expand moving elements to/from next bin ...
        needed_values = abs(count % -number_of_items_per_alignment)
        if needed_values:
            rest = counts[index + 1:]
            current_total_remainder = rest.sum()
            if needed_values <= (current_total_remainder + allotted_item_count):
                # we still have enough other values to work with, so expand ...
                counts[index] += needed_values  # safe to assume we will at least locate all needed items ...
                for i, s in enumerate(rest):
                    if s >= needed_values:
                        rest[i] -= needed_values
                        needed_values = 0
                    else:
                        needed_values -= s
                        rest[i] = 0
                    if not needed_values:  # done ...
                        break
                assert needed_values <= allotted_item_count
                allotted_item_count -= needed_values  # needed_values is either zero or non-zero <= allotted_item_count
            else:  # we don't have enough other values to work with, so check if we can shrink instead.
                if index == 0:
                    raise ValueError(  # if first element then we have no hope for a proper segmentation ... so quit.
                        'Unable to segment far too small array with {c} items of {b} bytes each over {a} bytes'.format(
                            c=flatten_a.size, b=flatten_a.itemsize, a=alignment
                        )
                    )
                excess = counts[index] % number_of_items_per_alignment
                counts[index + 1] = excess
                counts[index] -= excess

    # At this point all sizes except the very last one should be a multiple of number_of_items_per_alignment
    if counts[-1] % number_of_items_per_alignment:  # check last
        needed_count = abs(counts[-1] % -number_of_items_per_alignment)
        if needed_count <= allotted_item_count:
            counts[-1] += needed_count
        else:
            raise ValueError(
                'Unable to segment far too small array with {c} items of {b} bytes each over {a} bytes'.format(
                    c=flatten_a.size, b=flatten_a.itemsize, a=alignment
                )
            )
    # At this point all counts should be a multiple of number_of_items_per_alignment, create arrays ctype, calc addr
    starting_address = flatten_a.ctypes.data
    counts = sorted(counts, reverse=True)  # create larger arrays first ...
    return imap(
        lambda ct, index, sa=starting_address, i=flatten_a.itemsize: ct.from_address(sa + (index * i)),
        imap(numpy_type_to_ctypes[flatten_a.dtype.name].__mul__, sorted(counts, reverse=True)),  # get array ctypes ...
        chain((0,), imap(int, numpy.cumsum(counts))),  # offsets ...
    )

