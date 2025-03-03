# Use the NumPy mean() method to find the average speed:

#To calculate the mean, find the sum of all values, and divide the sum by the number of values:
import numpy
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = numpy.mean(speed)
print(x)
#Answer:    89.76923076923077 
print()

#The median value is the value in the middle, after you have sorted all the values:
import numpy
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = numpy.median(speed)
print(x)
#Answer:    87.0 
print()

#If there are two numbers in the middle, divide the sum of those numbers by two.
import numpy
speed = [99,86,87,88,86,103,87,94,78,77,85,86]
x = numpy.median(speed)
print(x)
#Answer:    86.5 
print()

# The Mode value is the value that appears the most number of times:
from scipy import stats
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = stats.mode(speed)
print(x)
#Answer:    ModeResult(mode=array([86]), count=array([3]))  
print()
#The mode() method returns a ModeResult object that contains the mode number (86), and count (how many times the mode number appeared (3)).


# Standard deviation is a number that describes how spread out the values are.
# A low standard deviation means that most of the numbers are close to the mean (average) value.
# A high standard deviation means that the values are spread out over a wider range.
import numpy
speed = [86,87,88,86,87,85,86]
x = numpy.std(speed)
print(x)
#Answer:    0.9035079029052513 
print()

import numpy
speed = [32,111,138,28,59,77,97]
x = numpy.std(speed)
print(x)
#Answer:    37.84501153334721 
print()

# (Variance) is another number that indicates how spread out the values are.
# In fact, if you take the square root of the variance, you get the standard deviation!
# Or the other way around, if you multiply the standard deviation by itself, you get the variance!
# 1. Find the mean:
# 2. For each value: find the difference from the mean:
# 3. For each difference: find the square value:
# 4. The variance is the average number of these squared differences:

#  (Standard Deviation = √Variance )
import numpy
speed = [32,111,138,28,59,77,97]
x = numpy.var(speed)
print(x)
#Answer:    1432.2448979591834 
#Standard Deviation = √1432.25 = 37.85 
print()

import numpy
speed = [32,111,138,28,59,77,97]
x = numpy.std(speed)
print(x)
#Answer:    37.85 
print()

# Standard Deviation is often represented by the symbol Sigma: σ
# Variance is often represented by the symbol Sigma Square: σ2 