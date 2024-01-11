import pytest
from uncertainties import ufloat, UFloat
from . uarray import UArray, unumpy_ufuncs
from uncertainties import unumpy, umath
import numpy as np

from uncertainties.test_uncertainties import numbers_close, arrays_close


@pytest.fixture
def data():
    return np.array([ufloat(x, 0.1) for x in [1, 2, 3]])

@pytest.mark.parametrize(
    "func",
    unumpy_ufuncs,
    )
def test_unumpy_ufuncs(data, func):
    numpy_func = getattr(np, func)
    unumpy_func = getattr(unumpy, func)
    umath_func = getattr(umath, func)

    # element-wise function application
    expected = np.array([umath_func(x) for x in data])
    
    # test UArray
    result = numpy_func(UArray(data))
    if isinstance(result, UArray):
        result = result.ndarray
    assert arrays_close(result, expected)

    # test uarray
    result = unumpy_func(data)
    assert arrays_close(result, expected)

