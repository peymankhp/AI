
import matplotlib
print(matplotlib.__version__)
print()
import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([0, 6])
ypoints = np.array([0, 250])
plt.plot(xpoints, ypoints)
plt.show() 
print()

# The plot() function is used to draw points (markers) in a diagram.
# By default, the plot() function draws a line from point to point.
# The function takes parameters for specifying points in the diagram.
# Parameter 1 is an array containing the points on the x-axis.
# Parameter 2 is an array containing the points on the y-axis.
# If we need to plot a line from (1, 3) to (8, 10), we have to pass two arrays [1, 8] and [3, 10] to the plot function.

import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints)
plt.show()
print()

import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])
plt.plot(xpoints, ypoints)
plt.show()

print()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10, 5, 7])
plt.plot(ypoints)
plt.show()

print()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o')
plt.show()

    # 'o' 	Circle 	
    # '*' 	Star 	
    # '.' 	Point 	
    # ',' 	Pixel 	
    # 'x' 	X 	
    # 'X' 	X (filled) 	
    # '+' 	Plus 	
    # 'P' 	Plus (filled) 	
    # 's' 	Square 	
    # 'D' 	Diamond 	
    # 'd' 	Diamond (thin) 	
    # 'p' 	Pentagon 	
    # 'H' 	Hexagon 	
    # 'h' 	Hexagon 	
    # 'v' 	Triangle Down 	
    # '^' 	Triangle Up 	
    # '<' 	Triangle Left 	
    # '>' 	Triangle Right 	
    # '1' 	Tri Down 	
    # '2' 	Tri Up 	
    # '3' 	Tri Left 	
    # '4' 	Tri Right 	
    # '|' 	Vline 	
    # '_' 	Hline

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, 'o:r')
plt.show()

print()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, 'o:r')
plt.show()

print()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o', ms = 20)
plt.show()

print()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r')
plt.show()

print()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o', ms = 20, mfc = 'r')
plt.show()

print()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r', mfc = 'r')
plt.show()

print()

...
plt.plot(ypoints, marker = 'o', ms = 20, mec = '#4CAF50', mfc = '#4CAF50')
... 
plt.plot(ypoints, marker = 'o', ms = 20, mec = 'hotpink', mfc = 'hotpink')

print()
