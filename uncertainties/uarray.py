from . core import ufloat, Variable, AffineScalarFunc, UFloat
# from . uarray_numpy_func import numpy_wrap
from uncertainties import umath_core as umath
from uncertainties import unumpy
import numpy as np

class UArray(np.lib.mixins.NDArrayOperatorsMixin):
    """ """

    def __init__(self, iterable):
        self.ndarray = np.array([ufloat(x)  if not isinstance(x, UFloat) else x for x in iterable
        ], dtype="object")

    def __len__(self):
        return len(self.ndarray)
    
    def __iter__(self):
        return iter(self.ndarray)

    def __getitem__(self, key):
        return self.ndarray[key]
    
    def __setitem__(self, key, value):
        if not isinstance(value, UFloat):
            value = ufloat(value)
        self.ndarray[key] = value

    def __repr__(self):
        return "<UArray " + str(self.ndarray) + " >"

    def __str__(self):
        return str(self.ndarray)
    
    def __array__(self):
        return self.ndarray

    @property
    def nominal_values(self):
        return np.array([x.n for x in self.ndarray])
    
    @property
    def std_devs(self):
        return np.array([x.s for x in self.ndarray])
    
    # NumPy function/ufunc support
    __array_priority__ = 17

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            # Only handle ufuncs as callables
            return NotImplemented

        # Replicate types from __array_function__
        types = {
            type(arg)
            for arg in list(inputs) + list(kwargs.values())
            if hasattr(arg, "__array_ufunc__")
        }

        return numpy_wrap("ufunc", ufunc, inputs, kwargs, types)

    def __array_function__(self, func, types, args, kwargs):
        return numpy_wrap("function", func, args, kwargs, types)

####################################################################################################

def is_upcast_type(t):
    return False

def implements(numpy_func_string, func_type):
    """Register an __array_function__/__array_ufunc__ implementation for UArray
    objects.

    """

    def decorator(func):
        if func_type == "function":
            HANDLED_FUNCTIONS[numpy_func_string] = func
        elif func_type == "ufunc":
            HANDLED_UFUNCS[numpy_func_string] = func
        else:
            raise ValueError(f"Invalid func_type {func_type}")
        return func

    return decorator

HANDLED_FUNCTIONS = {}
HANDLED_UFUNCS = {}

unumpy_ufuncs = [
 'arccos',
 'arccosh',
 'arcsin',
 'arctan',
 'arctan2',
 'arctanh',
 'asinh',
 'ceil',
 'copysign',
 'cos',
 'cosh',
 'degrees',
 'exp',
 'expm1',
 'fabs',
 'floor',
 'fmod',
 'hypot',
 'isinf',
 'isnan',
 'ldexp',
 'log',
 'log10',
 'log1p',
 'modf',
 'pow',
 'radians',
 'sin',
 'sinh',
 'sqrt',
 'tan',
 'tanh',
 'trunc',
]

# These functions are in unumpy, but are not ufuncs so need to be handled separately
#  'matrix',
#  'lgamma',
#  'gamma',
#  'erf',
#  'erfc',
#  'ulinalg',
#  'umatrix'
def wrap_result(result):
    if isinstance(result, tuple):
        return tuple(wrap_result(x) for x in result)
    if isinstance(result, np.ndarray) and result.dtype == "object" and any(isinstance(x, UFloat) for x in result):
        return UArray(result)
    else:
        return result

def unwrap_inputs(inputs):
    return tuple(x.ndarray if isinstance(x, UArray) else x for x in inputs)

def implement_unumpy_func(func_str):
    
    func = getattr(unumpy, func_str)
    @implements(func_str, "ufunc")
    def implementation(*inputs, **kwargs):
        inputs = unwrap_inputs(inputs)
        result = func(*inputs, **kwargs)
        return wrap_result(result)

for func in unumpy_ufuncs:
    implement_unumpy_func(func)

# https://numpy.org/doc/stable/reference/arrays.classes.html#numpy.class.__array_function__
def implement_unwrap_func(func_type, func_str):
    @implements(func_str, func_type)
    def implementation(*inputs, **kwargs):
        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.ndarray if isinstance(x, UArray) else x
                        for x in inputs)

        result = getattr(np, func_str)(*inputs, **kwargs)
        return UArray(result)
        # if type(result) is tuple:
        #     # multiple return values
        #     return tuple(type(self)(x) for x in result)
        # elif method == 'at':
        #     # no return value
        #     return None
        # else:
        #     # one return value
        #     return type(self)(result)
    return implementation

unwrap_ufuncs = [
    'add', 'subtract', 'multiply', 'divide', 'logaddexp', 'logaddexp2',
]
for func in unwrap_ufuncs:
    implement_unwrap_func('ufunc', func)

# https://stackoverflow.com/a/74137209/1291237
"""                                                                                                                                                                                                                                
Assuming Gaussian statistics, uncertainties stem from Gaussian parent distributions. In such a case,                                                                                                                               
it is standard to weight the measurements (nominal values) by the inverse variance.                                                                                                                                                
                                                                                                                                                                                                                                    
Following the pattern of np.mean, this function is really nan_mean, meaning it calculates based on non-NaN values.                                                                                                                 
If there are no such, it returns np.nan, just like np.mean does with an empty array.                                                                                                                                               
                                                                                                                                                                                                                                    
This function uses error propagation on the to get an uncertainty of the weighted average.                                                                                                                                         
:param: A set of uncertainty values                                                                                                                                                                                                
:return: The weighted mean of the values, with a freshly calculated error term                                                                                                                                                     
"""
@implements("mean", "function")
def mean(a, *args, **kwargs):
    if len(a) == 0:
        return np.nan
    if len(a) == 1:
        return a[0]

    nominals = a.nominal_values
    if any(a.std_devs == 0):
        # We cannot mix and match "perfect" measurements with uncertainties                                                                                                                                                            
        # Instead compute the mean and return the "standard error" as the uncertainty                                                                                                                                                  
        # e.g. ITR.umean([100, 200]) = 150 +/- 50                                                                                                                                                                                      
        w_mean = sum(nominals) / N
        w_std = np.std(nominals) / np.sqrt(N - 1)
    else:
        # Compute the "uncertainty of the weighted mean", which apparently                                                                                                                                                             
        # means ignoring whether or not there are large uncertainties                                                                                                                                                                  
        # that should be created by elements that disagree                                                                                                                                                                             
        # e.g. ITR.umean([100+/-1, 200+/-1]) = 150.0+/-0.7 (!)                                                                                                                                                                         
        w_sigma = 1 / sum([1 / (v.s**2) for v in a])
        w_mean = sum([v.n / (v.s**2) for v in a]) * w_sigma
        w_std = w_sigma * np.sqrt(sum([1 / (v.s**2) for v in a]))
    result = ufloat(w_mean, w_std)
    return result


def numpy_wrap(func_type, func, args, kwargs, types):
    """Return the result from a NumPy function/ufunc as wrapped by Pint."""

    if func_type == "function":
        handled = HANDLED_FUNCTIONS
        # Need to handle functions in submodules
        name = ".".join(func.__module__.split(".")[1:] + [func.__name__])
    elif func_type == "ufunc":
        handled = HANDLED_UFUNCS
        # ufuncs do not have func.__module__
        name = func.__name__
    else:
        raise ValueError(f"Invalid func_type {func_type}")

    if name not in handled or any(is_upcast_type(t) for t in types):
        return NotImplemented
    return handled[name](*args, **kwargs)