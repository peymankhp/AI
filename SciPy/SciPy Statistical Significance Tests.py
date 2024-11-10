# How to Implement it in SciPy?
# SciPy provides us with a module called scipy.interpolate which has many functions to deal with interpolation:
# 1D Interpolation
# The function interp1d() is used to interpolate a distribution with 1 variable.
# It takes x and y points and returns a callable function that can be called with new x and returns corresponding y.

# For given xs and ys interpolate values from 2.1, 2.2... to 2.9:
from scipy.interpolate import interp1d
import numpy as np
xs = np.arange(10)
ys = 2*xs + 1
interp_func = interp1d(xs, ys)
newarr = interp_func(np.arange(2.1, 3, 0.1))
print(newarr)
#Answer:   [ 5.2  5.4  5.6  5.8  6.   6.2  6.4  6.6  6.8]
print()

from scipy.interpolate import UnivariateSpline
import numpy as np
xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1
interp_func = UnivariateSpline(xs, ys)
newarr = interp_func(np.arange(2.1, 3, 0.1))
print(newarr) 
#Answer:     [5.62826474 6.03987348 6.47131994 6.92265019 7.3939103  7.88514634    8.39640439 8.92773053 9.47917082]
print()

from scipy.interpolate import Rbf
import numpy as np
xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1
interp_func = Rbf(xs, ys)
newarr = interp_func(np.arange(2.1, 3, 0.1))
print(newarr)
#Answer:     [6.25748981  6.62190817  7.00310702  7.40121814  7.8161443   8.24773402   8.69590519  9.16070828  9.64233874]
print()

# T-Test
# T-tests are used to determine if there is significant deference between means of two variables. and lets us know if they belong to the same distribution.
# It is a two tailed test.
# The function ttest_ind() takes two samples of same size and produces a tuple of t-statistic and p-value.

# Find if the given values v1 and v2 are from same distribution:
import numpy as np
from scipy.stats import ttest_ind
v1 = np.random.normal(size=100)
v2 = np.random.normal(size=100)
res = ttest_ind(v1, v2)
print(res)
#Answer:     Ttest_indResult(statistic=0.40833510339674095, pvalue=0.68346891833752133)
print()

# If you want to return only the p-value, use the pvalue property:
import numpy as np
from scipy.stats import ttest_ind
v1 = np.random.normal(size=100)
v2 = np.random.normal(size=100)
res = ttest_ind(v1, v2).pvalue
print(res)
#Answer:     0.370157125006
print()

import numpy as np
from scipy.stats import kstest
v = np.random.normal(size=100)
res = kstest(v, 'norm')
print(res)
#Answer:     KstestResult(statistic=0.084401539828641625, pvalue=0.45521368177068711)
print()

import numpy as np
from scipy.stats import describe
v = np.random.normal(size=100)
res = describe(v)
print(res)
 DescribeResult(
 #Answer:   nobs=100,
#     minmax=(-2.0991855456740121, 2.1304142707414964),
#     mean=0.11503747689121079,
#     variance=0.99418092655064605,
#     skewness=0.013953400984243667,
#     kurtosis=-0.671060517912661
#   )
print()

# Normality Tests (Skewness and Kurtosis)
# Normality tests are based on the skewness and kurtosis.
# The normaltest() function returns p value for the null hypothesis:
# "x comes from a normal distribution".
# Skewness:
# A measure of symmetry in data.
# For normal distributions it is 0.
# If it is negative, it means the data is skewed left.
# If it is positive it means the data is skewed right.
# Kurtosis:
# A measure of whether the data is heavy or lightly tailed to a normal distribution.
# Positive kurtosis means heavy tailed.
# Negative kurtosis means lightly tailed.

import numpy as np
from scipy.stats import skew, kurtosis
v = np.random.normal(size=100)
print(skew(v))
print(kurtosis(v)) 
#Answer:       0.11168446328610283      -0.1879320563260931
print()

#Find if the data comes from a normal distribution:
import numpy as np
from scipy.stats import normaltest
v = np.random.normal(size=100)
print(normaltest(v)) 
#Answer:       NormaltestResult(statistic=4.4783745697002848, pvalue=0.10654505998635538)
print()
















