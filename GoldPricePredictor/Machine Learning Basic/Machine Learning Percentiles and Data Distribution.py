# What are Percentiles?
# Percentiles are used in statistics to give you a number that describes the value that a given percent of the values are lower than.
# Example: Let's say we have an array of the ages of all the people that lives in a street.

import numpy
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = numpy.percentile(ages, 75)
print(x)
#Answer:   43.0 
print()

import numpy
ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = numpy.percentile(ages, 90)
print(x)
#Answer:   61.0  
print()

# Data Distribution
# Earlier in this tutorial we have worked with very small amounts of data in our examples, just to understand the different concepts.
# In the real world, the data sets are much bigger, but it can be difficult to gather real world data, at least at an early stage of a project.

#Create an array containing 250 random floats between 0 and 5:
import numpy
x = numpy.random.uniform(0.0, 5.0, 250)
print(x)
#Answer:   [2.51914675 4.25593242 1.70996888 2.59532988 1.8764142  1.92012275  2.91690949 0.52829797 4.79250388 2.89169523]
print()

#Create an array containing 250 random floats between 0 and 5: (Histogram)
import numpy
import matplotlib.pyplot as plt
x = numpy.random.uniform(0.0, 5.0, 25)
plt.hist(x, 5)
plt.show() 
# Histogram Explained
# Which gives us this result:
#     52 values are between 0 and 1
#     48 values are between 1 and 2
#     49 values are between 2 and 3
#     51 values are between 3 and 4
#     50 values are between 4 and 5
print()

# Create an array with 100000 random numbers, and display them using a histogram with 100 bars:
import numpy
import matplotlib.pyplot as plt
x = numpy.random.uniform(0.0, 5.0, 100000)
plt.hist(x, 100)
plt.show() 
print()


# Normal Data Distribution
# In the previous chapter we learned how to create a completely random array, of a given size, and between two given values.
# In this chapter we will learn how to create an array where the values are concentrated around a given value.
# In probability theory this kind of data distribution is known as the normal data distribution, or the Gaussian data distribution,
#  after the mathematician Carl Friedrich Gauss who came up with the formula of this data distribution.

import numpy
import matplotlib.pyplot as plt
x = numpy.random.normal(5.0, 1.0, 100000)
plt.hist(x, 100)
plt.show() 
print()
# We use the array from the numpy.random.normal() method, with 100000 values,  to draw a histogram with 100 bars.
# We specify that the mean value is 5.0, and the standard deviation is 1.0.
# Meaning that the values should be concentrated around 5.0, and rarely further away than 1.0 from the mean.
# And as you can see from the histogram, most values are between 4.0 and 6.0, with a top at approximately 5.0.

#Scatter Plot

# The Matplotlib module has a method for drawing scatter plots, 
# it needs two arrays of the same length, one for the values of the x-axis, and one for the values of the y-axis:

import matplotlib.pyplot as plt
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x, y)
plt.show()

# Let us create two arrays that are both filled with 1000 random numbers from a normal data distribution.
# The first array will have the mean set to 5.0 with a standard deviation of 1.0.
# The second array will have the mean set to 10.0 with a standard deviation of 2.0:

import numpy
import matplotlib.pyplot as plt
x = numpy.random.normal(5.0, 1.0, 1000)
y = numpy.random.normal(10.0, 2.0, 1000)
plt.scatter(x, y)
plt.show()
print()

# Linear Regression
# The term regression is used when you try to find the relationship between variables.
# In Machine Learning, and in statistical modeling, that relationship is used to predict the outcome of future events.
# Linear regression uses the relationship between the data-points to draw a straight line through all them.
# This line can be used to predict future values.

# In the example below, the x-axis represents age, and the y-axis represents speed. 
# We have registered the age and speed of 13 cars as they were passing a tollbooth. Let us see if the data we collected could be used in a linear regression:
import matplotlib.pyplot as plt
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x, y)
plt.show() 
print()


import matplotlib.pyplot as plt
from scipy import stats
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show() 
# mport the modules you need.
# You can learn about the Matplotlib module in our Matplotlib Tutorial.
# You can learn about the SciPy module in our SciPy Tutorial.
# import matplotlib.pyplot as plt
# from scipy import stats
# Create the arrays that represent the values of the x and y axis:
# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# Execute a method that returns some important key values of Linear Regression:
# slope, intercept, r, p, std_err = stats.linregress(x, y)
# Create a function that uses the slope and intercept values to return a new value. This new value represents where on the y-axis the corresponding x value will be placed:
# def myfunc(x):
#   return slope * x + intercept
# Run each value of the x array through the function. This will result in a new array with new values for the y-axis:
# mymodel = list(map(myfunc, x))
# Draw the original scatter plot:
# plt.scatter(x, y)
# Draw the line of linear regression:
# plt.plot(x, mymodel)
# Display the diagram:
print()

#How well does my data fit in a linear regression?
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)
#Answer:   -0.758591524376155 
print()

# Predict Future Values
# Now we can use the information we have gathered to predict future values.
# Example: Let us try to predict the speed of a 10 years old car.
# To do so, we need the same myfunc() function from the example above:
from scipy import stats
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
  return slope * x + intercept
speed = myfunc(10)
mymodel = list(map(myfunc, x))
print(speed) 
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show() 
#Answer:   85.59308314937454 
print()

#Let us create an example where linear regression would not be the best method to predict future values.
#These values for the x- and y-axis should result in a very bad fit for linear regression:
import matplotlib.pyplot as plt
from scipy import stats
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
  return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
print()

#You should get a very low r value.
import numpy
from scipy import stats
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)
#Answer:   0.01331814154297491
print()