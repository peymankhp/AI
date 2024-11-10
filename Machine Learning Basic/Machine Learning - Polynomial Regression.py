#Polynomial Regression
#If your data points clearly will not fit a linear regression (a straight line through all data points), it might be ideal for polynomial regression.
#Polynomial regression, like linear regression, uses the relationship between the variables x and y to find the best way to draw a line through the data points.
import matplotlib.pyplot as plt
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
plt.scatter(x, y)
plt.show() 

#Import numpy and matplotlib then draw the line of Polynomial Regression:
# Example Explained
# Import the modules you need.
# You can learn about the NumPy module in our NumPy Tutorial.
# You can learn about the SciPy module in our SciPy Tutorial.
import numpy
import matplotlib.pyplot as plt
# Create the arrays that represent the values of the x and y axis:
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
# NumPy has a method that lets us make a polynomial model:
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
# Then specify how the line will display, we start at position 1, and end at position 22:
myline = numpy.linspace(1, 22, 100)
# Draw the original scatter plot:
plt.scatter(x, y)
# Draw the line of polynomial regression:
plt.plot(myline, mymodel(myline))
# Display the diagram:
plt.show()
print()


# R-Squared
# It is important to know how well the relationship between the values of the x- and y-axis is, if there are no relationship the polynomial regression can not be used to predict anything.
# The relationship is measured with a value called the r-squared.
# The r-squared value ranges from 0 to 1, where 0 means no relationship, and 1 means 100% related.
# Python and the Sklearn module will compute this value for you, all you have to do is feed it with the x and y arrays:
import numpy
from sklearn.metrics import r2_score
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
print(r2_score(y, mymodel(x)))
#Answer:    0.9432150416451027 
print()


# Predict Future Values
# Now we can use the information we have gathered to predict future values.
# Example: Let us try to predict the speed of a car that passes the tollbooth at around 17 P.M:
# To do so, we need the same mymodel array from the example above:
import numpy
from sklearn.metrics import r2_score
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
speed = mymodel(17)
print(speed)
#Answer:    88.87331269697987 
print()

# Bad Fit?
# Let us create an example where polynomial regression would not be the best method to predict future values.
import numpy
import matplotlib.pyplot as plt
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(2, 95, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show() 
#Answer:    88.87331269697987 
print()

import numpy
from sklearn.metrics import r2_score
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
print(r2_score(y, mymodel(x)))
#Answer:    0.009952707566680652 
print()
