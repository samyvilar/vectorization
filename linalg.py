__author__ = 'samyvilar'

import struct
import numpy

from itertools import chain, izip, imap, ifilter, starmap, repeat
import scipy
import scipy.optimize

get_max_float_type = lambda t: numpy.float128 if getattr(t, 'dtype', float) is numpy.float128 else numpy.float


__HI = lambda v: struct.unpack('i', struct.pack('d', v)[4:])[0]  # assuming little endian ...
interpret_double_as_long = lambda d: struct.unpack('q', struct.pack('d', d))[0]
interpret_long_as_double = lambda l: struct.unpack('d', struct.pack('q', l))[0]
double_to_byte_array = lambda v: struct.pack('d', v)[4:]
double_to_bit_array = lambda v: '{0:064b}'.format(interpret_double_as_long(v))


def log2_through_poly_fit(value, poly_degree=8, poly_error=2**-51):
    """
        IEEE double precision is represented as:
        value = (-1**b63)(2**(exponent-1023))(1 + sum(i=1..52, b(52-i)(2**-i)) where b is 63..0
        therefore:
            log2(value) = log2((-1**b63)(2**(exponent-1023))(1 + sum(i=1..52, b(52-i)(2**-i)))
        apply log rule log(ab) = log(a) + log(b)
            log2(value) = log2(-1**b63) + log2(2**(exponent-1023)) + log2(1 + sum(i=1..52, b(52-i)(2**-i)))
            log2(1) = 0 log2(-1) nan (undefined) ...
            log2(2**x) = x
        so:
            log2(value) = 0 + (exponent - 1023) + log2(1 + sum(i=1..52, b(52-i)(2**-i)))
            log2(value) = exponent - 1023 + log2(1 + sum(i=1..52, b(52-i)(2**-i)))
            1 <= 1 + sum(i=1..52, b(52-i)(2**-i)) < 2, so we need to a polyfit of log2 between 1 and 2
        Note that in this case decreasing the interval has no positive effect (accuracy improvement),
        it actually has a negative effect (accuracy worsens)!
    """
    interval = (1.0, 2.0)
    coefficients = remez(lambda v: numpy.log2(v)/(v - 1.0), interval, poly_degree, error=poly_error)
    double_as_long = interpret_double_as_long(value)
    exponent = ((double_as_long & (0x7FF << 52)) >> 52)
    # get mantissa add one to it, to get the actual value
    mantissa = interpret_long_as_double((double_as_long & ((1 << 52) - 1)) | interpret_double_as_long(1.0))
    return exponent - 1023 + (numpy.polyval(coefficients, mantissa) * (mantissa - 1.0))


def exp_ieee(x):
    ln2HI = 6.93147180369123816490e-01, -6.93147180369123816490e-01
    ln2LO = 1.90821492927058770002e-10, -1.90821492927058770002e-10
    invln2 = 1.44269504088896338700e+00
    halF = 0.5, -0.5
    one = 1.0
    huge = 1.0e+300
    twom1000 = 9.33263618503218878990e-302
    P1 = 1.66666666666666019037e-01
    P2 = -2.77777777770155933842e-03
    P3 = 6.61375632143793436117e-05
    P4 = -1.65339022054652515390e-06
    P5 = 4.13813679705723846039e-08
    hx = __HI(x)  # high word of double interpreted as a 32 bit int
    xsb = (hx >> 31) & 1    # sign bit of x
    hx &= 0x7fffffff    # /* high word of |x| removing sign bit */
    # /* argument reduction */
    if hx > 0x3fd62e42:  # { /* if  |x| > 0.5 ln2 */
        if hx < 0x3FF0A2B2:  # {	/* and |x| < 1.5 ln2 */
            hi = x - ln2HI[xsb]
            lo = ln2LO[xsb]
            k = 1 - xsb - xsb
        else:
            k = int(invln2 * x + halF[xsb])
            t = k
            hi = x - t * ln2HI[0]   # /* t*ln2HI is exact here */
            lo = t * ln2LO[0]   # }
        x = hi - lo
    elif hx < 0x3e300000:  # when |x|<2**-28
        if huge + x > one:
            return one + x  # /* trigger inexact */ #}
    else:
        k = 0
    # /* x is now in primary range */
    t = x * x
    c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))))
    if k == 0:
        return one - ((x * c) / (c - 2.0) - x)
    else:
        y = one - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= -1021:
        # y = __HI(y) + (k << 20)
        # __HI(y) += (k << 20)  # /* add k to y's exponent */
        return struct.unpack('d', struct.pack('d', y)[0:4] + struct.pack('i', __HI(y) + (k << 20)))[0]
    else:
        # y = __HI(y) + ((k + 1000) << 20)
        y = struct.unpack('d', struct.pack('d', y)[0:4] + struct.pack('i', ((k + 1000) << 20)))[0]
        # __HI(y) += ((k+1000)<<20)  # /* add k to y's exponent */
        return y * twom1000


def exp_through_polynomial_fit(value, length=12):
    """
        http://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
        assuming powf(x, y) == exp2(log2(x) * y)) since x**y = 2**(log2(x**y)) = 2**(y * log2(x))
        then powf(e, y) == exp2(log2(e) * y)), log2(e) = 1.442695040888963387004650940071,
        exp2(1.442695040888963387004650940071 * y)
        break apart (1.442695040888963387004650940071 * y) into real and integral,

        IEEE doubles are represented using 64 bits where:
            value = -1**b[63] + (int(b[52:64]) - 1023) + 1 + sum(b[52 - i]/2**i for i in xrange(52))

        since x**(real + integral) => x**real * x**integral
        implement the integral part using fast shifts,
        the real portion will be implemented using a polynomial function ...
        we can further increase the accuracy by reducing the interval from (-1, 1) to (-.5, .5) by:
        taking the square root of each side and then squaring the final answer, Proof:
        (e**x)**0.5 = (2**(x * log2(e)))**0.5, let y = x * log2(e)
        (2**y)**0.5 = (2**(floor(y) + (y - floor(y))))**0.5 = (2**(floor(y)))**05 * (2**(y - floor(y)))**0.5
        (2**(y - floor(y)))**0.5 = 2**(0.5 * (y - floor(y))
        since -1 < y - floor(y) < 1 we have -0.5 < 0.5 * (y - floor(y)) < 0.5
        the final result would simply need to be squared since ((e**x)**0.5)**2 = (e**x)**(2*0.5) = e**x ...
    """
    y = value * 1.442695040888963387004650940071
    integral = numpy.sqrt(numpy.exp2(int(y)))
    return (integral * numpy.polyval(remez(numpy.exp2, (-0.5, 0.5), length), (y - int(y))/2.0))**2


def brentq(f, a, b, args=(), xtol=None, rtol=4.4408920985006262e-16, maxiter=100):
    fa, fb = imap(f, (a, b))
    assert fa * fb <= 0
    a, b = (b, a) if numpy.abs(fa) < numpy.abs(f(b)) else (a, b)
    c = a
    d = None
    fc = fa
    flag = True
    conditions = (
        lambda s, flag, a, b, d: not (((3*a + b) / 4) < s < b or ((3 * a + b) / 4) < s < b),
        lambda s, flag, a, b, d: flag and numpy.abs(s - b) >= (numpy.abs(b - c) / 2),
        lambda s, flag, a, b, d: not flag and numpy.abs(s - b) >= (numpy.abs(c - d) / 2),
        lambda s, flag, a, b, d: flag and numpy.abs(b - c) < numpy.abs(xtol),
        lambda s, flag, a, b, d: not flag and numpy.abs(c - d) < numpy.abs(xtol)
    )
    xtol = numpy.finfo(type(b)).eps if xtol is None else xtol
    while fb and (numpy.abs(a - b) > xtol) and maxiter:
        s = (a * fb * fc / (fa - fb) / (fa - fc) + b * fa * fc / (fb - fa) / (fb - fc) + c * fa * fb / (fc - fa) /
             (fc - fb)) if fa != fc and fb != fc else (b - fb * (b - a) / (fb - fa))  # Secant Rule
        flag = any(c(s, flag, a, b, d) for c in conditions)
        s = ((a + b) / 2) if flag else s  # bisection method ...
        fs = f(s)
        d = c
        c = b
        if (fa * fs) < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        a, b = (b, a) if numpy.abs(f(a)) < numpy.abs(f(b)) else (a, b)
        maxiter -= 1
        if not maxiter:
            print 'error: {0}'.format(fb)
    return b


def remez_get_extremas(func, interval, extremas=None, roots=None, accuracy=None):
    # Get all local minimums and maximums between the interval
    # only applicable to oscillating polynomial functions ie error function of the remez algorithm ...
    # hence the need for the current extremas
    # method 1: (simple but incredibly inefficient) take the x values at which continuous difference change in sign.
    #   it does not require any knowledge on the giving function besides the end points.
    #   xvalues = numpy.linspace(*chain(end_points, (10**5,),))
    #   return numpy.asarray(tuple(
    #         chain(
    #             (xvalues[0],),
    #           # locate the extremas by the location at which deltas change signs ...
    #           xvalues[numpy.where(numpy.diff(numpy.sign(numpy.diff(func(xvalues)))) != 0)],
    #           (xvalues[-1],)
    #       )
    #   ))
    # method 2: (faster than method 1 but not very accurate) interpolate the error function over degree + 1
    #   polynomial, take the roots of the derivative of this polynomial
    #   (this are the 'rough estimates' of the minimums & maximums)...
    #   # interpolate error function over a polynomial of degree + 1
    #   poly_fit_error_func = numpy.polyfit(xvalues, func(xvalues), number_of_roots)
    #   return (chain((xvalues[0],), sorted(numpy.roots(numpy.polyder(poly_fit_error_func))), (xvalues[-1],))))
    # method 3: (locate roots) assuming they are 'initially' each evenly apart
    #   (this may actually may be ok for the remez algorithm) ==> [scratch that! it picks adjacent non-min-max values]
    #   using the located roots as pairs find all the minimums and (maximums by negation of func) using Brent algorithm
    # method 4: Brent's algorithm seems to be the fastest ...
    min_max_fs = func, lambda v: numpy.negative(func(v))  # negate to find maximum ...
    assert (extremas is not None) ^ (roots is not None)  # make sure either extremas or roots are giving but not both
    # find roots using extremas if they weren't supplied ... scipy.optimize.brentq
    brentq_func = None
    if roots is None:
        brentq_func = brentq if get_max_float_type(extremas[0]) == numpy.float128 else scipy.optimize.brentq
    roots = starmap(brentq_func, izip(repeat(func), extremas, extremas[1:])) if roots is None else roots
    locations = tuple(chain((interval[0] - numpy.finfo(float).eps,), roots, (interval[1] + numpy.finfo(float).eps,)))
    accuracy = numpy.finfo(type(locations[1])).eps if accuracy is None else accuracy
    return numpy.fromiter(
        starmap(
            scipy.optimize.fminbound,
            izip(  # cycle through minimization/maximization, depending on initial direction ...
                cycle(min_max_fs if func(interval[0]) - func(interval[0] + accuracy) < 0 else reversed(min_max_fs)),
                locations,          # lower bounds ...
                locations[1:],      # upper bounds
                repeat(()),         # args to minimization func ...
                repeat(accuracy)    # tolerance
            )
        ),
        locations[1].dtype,
        len(locations) - 1
    )

# $ python -m timeit "one_if_even_negative_one_if_odd = lambda x: -1 if x % 2 else 1;
#   _ = map(one_if_even_negative_one_if_odd, xrange(100000))"
# 10 loops, best of 3: 23.6 msec per loop
# $ python -m timeit "one_if_even_negative_one_if_odd = lambda x: (-2 * (x % 2)) + 1;
#   _ = map(one_if_even_negative_one_if_odd, xrange(100000))"
# 10 loops, best of 3: 27.9 msec per loop
# $ python -m timeit "one_if_even_negative_one_if_odd = lambda x: (-1)**x;
#   _ = map(one_if_even_negative_one_if_odd, xrange(100000))"
# 10 loops, best of 3: 43.6 msec per loop
alternating_signs = lambda shape: numpy.fromiter(  # create a matrix of alternating 1 and -1 ...
    ((-1 if numpy.sum(indices) % 2 else 1) for indices in product(*imap(xrange, shape))), 'int', numpy.product(shape)
    # (-1)**(numpy.sum(indices))
).reshape(shape)

iminor = lambda m, i, j: imap(  # return all values where column_index != j
    lambda enum_row, j=j: imap(
        tuple.__getitem__,  # get the actual value ... (remove index)
        ifilter(lambda enum_col, j=j: enum_col[0] != j, enumerate(enum_row[1])),
        repeat(1)
    ),
    ifilter(lambda args, i=i: args[0] != i, enumerate(m))  # return all rows where row_index != i
)

is_square_matrix = lambda m: (len(m.shape) == 2 and m.shape[0] == m.shape[1]) if hasattr(m, 'shape') \
    else {len(m)} == set(len(e) for e in m)


# noinspection PyTypeChecker
# noinspection PyNoneFunctionAssignment
def doolittle(m):
    assert is_square_matrix(m)
    matrix_dim, dtype = len(m), get_max_float_type(m)
    pivot_vector = numpy.arange(matrix_dim)
    lower, upper = numpy.identity(matrix_dim, dtype=dtype), numpy.copy(m).astype(dtype)
    for k in xrange(matrix_dim - 1):
        max_row = numpy.argmax(numpy.abs(upper[k:, k])) + k
        if max_row != k:  # pivot rows if the current diagonal element isn't the largest in magnitude ...
            temp, upper[k, k:] = numpy.copy(upper[k, k:]), upper[max_row, k:]
            upper[max_row, k:] = temp
            temp, lower[k, :k] = numpy.copy(lower[k, :k]), lower[max_row, :k]
            lower[max_row, :k] = temp
            pivot_vector[k], pivot_vector[max_row] = pivot_vector[max_row], pivot_vector[k]
        lower[k + 1:, k] = upper[k + 1:, k] / upper[k, k]  # for j = k + 1 : m   L(j, k) = U(j, k)/U(k, k)
        upper[k + 1:, k:] = upper[k + 1:, k:] - numpy.dot(lower[k + 1:, k:k+1], upper[k:k+1, k:])
    return numpy.identity(matrix_dim)[pivot_vector], lower, upper


# noinspection PyTypeChecker
def det(m):
    """
        det(A) = det(inv(P)) * det(L) * det(U), where dot(P, A) == dot(L, U)
        the determinant of a triangular matrix m is the product(diag(m))
        note: P^-1 == P.T
    """
    assert is_square_matrix(m)
    if len(m) == 2:
        return m[0][0]*m[1][1] - m[1][0]*m[0][1]
    p, _, u = doolittle(m)      # numpy.product(numpy.diag(l)) == 1
    return numpy.linalg.det(p.T) * numpy.product(numpy.diag(u))

# naive approach incredibly slow n! running time!
# determinant = lambda m: (m[0, 0] * m[1, 1]) - (m[0, 1] * m[1, 0]) if m.shape == (2, 2) else sum(  # TopDown
#   # calculate matrix determinant ...# (-1)**col_index
#   ((-1 if col_index % 2 else 1) * value * determinant(minor(m, 0, col_index)) for col_index, value in enumerate(m[0]))
# )
#

minor = lambda m, i, j: m[  # get minor matrix, (all the values excluding those at row i and column j of matrix m.)
    numpy.fromiter(   # numpy.array(range(i) + range(i+1, m.shape[0]))[:, numpy.newaxis],
        chain(xrange(i), xrange(i + 1, m.shape[0])),
        'int',
        count=m.shape[0] - 1
    )[:, numpy.newaxis],
    numpy.fromiter(  # numpy.array(range(j) + range(j + 1, m.shape[1]))
        chain(xrange(j), xrange(j + 1, m.shape[1])),
        'int',
        count=m.shape[1] - 1
    )
]


def bi_directional_substitution(fx, pivot, lower, upper):
    """
        return x such that dot(lower, upper, x) =~= dot(pivot, fx) using forward and backwards substitution.

        informal proof:
            Assume that:
            - lower and upper are a lower unit and upper triangular square matrices
            - pivot is a row permutation of the identity matrix.
            - dot(lower, upper, x) =~= dot(pivot, fx) for some unknown vector x,
            - let n = len(fx) - 1 then:
        -> dot(lower, upper, x) =~= dot(pivot, fx)
        -> dot(lower, y) =~= dot(pivot, fx) where y = dot(upper, x), we use forward substitution to locate y:
            -> dot(lower, y) =~= dot(pivot, fx) == [sum(lower[n, :], y) for n in xrange(len(lower))]
            -> y[0] == dot(pivot, fx)[0], since lower is unit triangular matrix
            -> dot(pivot, fx)[k] == dot(lower[k, :], y) == sum(lower[k, :] * y[:])
            -> sum(lower[k, :] * y[:]) == lower[k, 0] * y[0] + lower[k, 1] * y[1] + ... + lower[n, k] * y[n],
             but since lower is a lower triangular matrix:
            -> dot(pivot, fx)[k] for some k <= n
                == sum(lower[k, :] * y[:])
                == (lower[k, 0] * y[0] + lower[k, 1] * y[1] + ... + lower[k - 1, k] * y[k - 1]) + lower[k, k] * y[k]
            -> y[k] =~= (dot(pivot, fx)[k] - dot(lower[:k - 1, :], y[:k - 1])) / lower[k, k]

        -> dot(upper, x) = y, use backward substitution to locate x, this is the 'approximate' solution...
        -> y[n] == upper[n, n] * x[n], since upper is an upper triangular matrix
        -> x[n] =~= y[n] / upper[n, n]
        -> y[k] == dot(upper[k, :], x) == sum(upper[k, :] * x[:]) but since upper is an upper triangular matrix:
        -> y[k] == sum(upper[k, k:], x[k:]) == upper[k, k] * x[k] + upper[k, k + 1] * x[k + 1] + .. + upper[k, n] * x[n]
        -> x[k] =~= (y[k] - dot(upper[k + 1:, x[k + 1:])) / upper[k, k]
    """
    pivot_fx = numpy.dot(pivot, fx)
    y, x = numpy.zeros((2, len(fx)), dtype=fx.dtype)
    for k, row in enumerate(lower):  # solve for y using forward substitution
        y[k] = (pivot_fx[k] - numpy.dot(row[:k], y[:k])) / row[k]
    for k, row in izip(*imap(reversed, (xrange(len(fx)), upper))):  # solve for x using backwards substitution
        x[k] = (y[k] - numpy.dot(row[k + 1:], x[k + 1:])) / row[k]
    return x


# noinspection PyNoneFunctionAssignment
# noinspection PyTypeChecker
def solve(square_x, fx):
    """
        solve a system of linear equations using lup decomposition.
            returns a vector x such that dot(square_x, x) =~= fx

        informal proof:
        giving the square matrix square_x and vector fx of same length as square_x,
            with the calculated square matrices: pivot, lower, upper where:
            - lower and upper are lower and upper triangular matrices accordingly
            - pivot is a row permutation of the identity matrix
            - =~= represents almost identical value giving some round off error
            - dot(*args) represents the dot product (matrix multiplication) of 2 more arguments
                where dot(A, B, C, ...) is interpreted as dot(dot(dot(A, B), C), ...) and is associative.
            - inv(arg) is the matrix inversion operator such that for square args
                dot(inv(arg), arg) = I = dot(arg, inv(arg))
            - dot(pivot, square_x) =~= dot(lower, upper) and dot(square_x, x) =~= fx
        then:
        -> square_x =~= dot(inv(pivot), lower, upper)
        -> dot(inv(pivot), lower, upper, x) =~= fx
        -> dot(pivot, inv(pivot), lower, upper, x) =~= dot(pivot, fx),
        -> dot(lower, upper, x) =~= dot(pivot, fx), use bi_directional_substitution (forward/backwards) to solve for x
    """
    return bi_directional_substitution(fx, *doolittle(square_x))


def inv(square_x):
    """
        find the multiplicative inverse of the square matrix square_x using lup decomposition.
            returns a matrix inverse_x such that dot(inverse_x, square_x) =~= I =~= dot(square_x, inverse_x)

        note: if the intention is to solve a system of linear Ax = b by x =~= dot(inv(A), b)
            equations use solve instead for better accuracy.

        informal proof:
            giving the square matrices square_x and inverse_x, calculate pivot, lower upper such that:
                - dot(pivot, square_x) =~= dot(lower, upper) and dot(square_x, inverse_x) = I then:
        -> square_x =~= dot(inv(pivot), lower, upper)
        -> dot(square_x, inverse_x) =~= I
        -> dot(inv(pivot), lower, upper, inverse_x) =~= I
        -> dot(pivot, inv(pivot), lower, upper, inverse_x) =~= dot(pivot, I) == pivot
        -> dot(lower, upper, inverse_x) =~= pivot
        -> dot(lower, upper, inverse_x[:, k]) =~= pivot[:, k]
            use bi_directional_substitution which returns x such that dot(lower, upper, x) =~= dot(pivot, fx)
                to calculate each of the columns of inverse_x:
    """
    pivot, lower, upper = doolittle(square_x)
    n = len(square_x)
    identity = numpy.identity(n)
    return numpy.fromiter(  # solve for each column of X
        chain.from_iterable(bi_directional_substitution(column, identity, lower, upper) for column in pivot.T),
        get_max_float_type(square_x[0][0]),
        count=n**2,
    ).reshape((n, n)).T  # numpy.fromiter reads row by row so we need to Transpose to convert them to columns ...


# naive implementation, very slow!
# def inv(square_matrix):  # matrix inversion, numpy.linalg.inv only supports float64, this should work with float128
#     size, shape, element_type = imap(getattr, repeat(square_matrix), ('size', 'shape', 'dtype'))
#     matrix_of_minors = numpy.fromiter(
#         (det(minor(square_matrix, row_index, col_index))
#          for row_index, row in enumerate(square_matrix) for col_index, value in enumerate(row)),
#         element_type,
#         count=size
#     ).reshape(shape)
#     co_factors = matrix_of_minors * alternating_signs(shape)
#     return co_factors.T / numpy.float128((square_matrix[0, :] * co_factors[0, :]).sum())


def remez(func, interval, degree, error=None, maxiter=30, float_type=numpy.float128):
    """
        The remez algorithm is an iterative algorithm for finding the optimal polynomial for a giving function on a
    closed interval.
        Chebyshev showed that such a polynomial 'exists' and is 'unique', and meets the following:
            - If R(x) is a polynomial of degree N, then there are N+2 unknowns:
                the N+1 coefficients of the polynomial, and maximal value of the error function.
            - The error function has N+1 roots, and N+2 extrema (minima and maxima).
            - The extrema alternate in sign, and all have the same magnitude.
        The key to finding this polynomial is locating those locations withing then closed interval, that meets all
        three of these properties.
    If we know the location of the extrema of the error function, then we can write N+2 simultaneous equations:
        R(xi) + (-1)iE = f(xi)
    where E is the maximal error term, and xi are the abscissa values of the N+2 extrema of the error function.
    It is then trivial to solve the simultaneous equations to obtain the polynomial coefficients and the error term.
    Unfortunately we don't know where the extrema of the error function are located!

    The remez method is used to locate (hopefully converge in a timely manner) on such locations.

    1) Start by a 'good' estimate, using Chebyshev roots as the points in questions.
    note: this are only applicable on the interval [-1, 1], hence the Chebyshev roots need to be linearly mapped
        to the giving interval [a, b].
    2) Using polynomial interpolation or any other method to locate the initial set of coefficients ...
    3) Locate all local extrema there should N+2 such locations see: get_extrema
    4) create a new solution, (coefficients + error_term) using the extrema(s), if the error_term doesn't change
        by a certain amount quit since progress can no long er be made
        otherwise use the previous extrema(s) as the new locations and repeat steps 3, 4 ...
    """
    f = func if type(func) is numpy.ufunc else numpy.vectorize(func)  # vectorized non-numpy functions ...
    # numpy.pi is a float64 value, this should give us a bit more accuracy ...
    one, two, four, five, sixteen = imap(float_type, (1, 2, 4, 5, 16))
    pi = sixteen * numpy.arctan(one / five) - four * numpy.arctan(one / float_type(239))
    chebyshev_nodes = numpy.cos(  # locate all needed chebyshev nodes ...
        (((two * degree + one - two * numpy.arange(0, degree + 1, dtype=float_type)) * pi)/(two * degree + two))
    )
    # linearly map chebyshev nodes from (-1, 1) to the giving interval, scale + offset ...
    x = (numpy.diff(interval) / two) * chebyshev_nodes + numpy.sum(interval) / two
    fx = f(x)
    coefficients = solve(numpy.vander(x), fx)  # solve the system ...
    # relative error function .. bind the current coefficients to it ...
    rel_error_func = lambda v, coefficients=coefficients, f=f: (numpy.polyval(coefficients, v) - f(v))/f(v)
    alternating_sign = alternating_signs((degree + 2,))
    delta_error_term, error_term = 10, 1000
    x = remez_get_extremas(rel_error_func, interval, roots=x)  # get extremas from Chebyshev roots and use them for sol
    error = numpy.finfo(x.dtype).eps if error is None else error  # set the error to the floats machine epsilon ...
    while abs(delta_error_term) > error and maxiter:  # continue while making progress
        x = remez_get_extremas(
            lambda v, coefficients=coefficients, f=f: rel_error_func(v, coefficients, f), interval, x, accuracy=error
        )
        fx = f(x)
        new_solution = solve(  # solve the system of N + 2 equations to get a new solution and error term
            numpy.append(numpy.vander(x, degree + 1), (alternating_sign * numpy.abs(fx)).reshape(-1, 1), axis=1), fx
        )  # I think f(xi)*-1**i has to be added as the last term (E) in order for errorfunc to equioscillate at extrema
        delta_error_term = new_solution[-1] - error_term
        coefficients, error_term = new_solution[:-1], new_solution[-1]
        maxiter -= 1
    return coefficients
