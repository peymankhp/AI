import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
points = np.array([
  [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1]
])
simplices = Delaunay(points).simplices
plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')
plt.show()
print()

# Convex Hull
# A convex hull is the smallest polygon that covers all of the given points.
# Use the ConvexHull() method to create a Convex Hull.

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
points = np.array([
  [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1],
  [1, 2],
  [5, 0],
  [3, 1],
  [1, 2],
  [0, 2]
])
hull = ConvexHull(points)
hull_points = hull.simplices
plt.scatter(points[:,0], points[:,1])
for simplex in hull_points:
  plt.plot(points[simplex,0], points[simplex,1], 'k-')
plt.show()
print()

# KDTrees
# KDTrees are a datastructure optimized for nearest neighbor queries.
# E.g. in a set of points using KDTrees we can efficiently ask which points are nearest to a certain given point.
# The KDTree() method returns a KDTree object.
# The query() method returns the distance to the nearest neighbor and the location of the neighbors.

from scipy.spatial import KDTree
points = [(1, -1), (2, 3), (-2, 3), (2, -3)]
kdtree = KDTree(points)
res = kdtree.query((1, 1))
print(res)
#Answer:  (2.0, 0)  
print()

# Find the euclidean distance between given points.
from scipy.spatial.distance import euclidean
p1 = (1, 0)
p2 = (10, 2)
res = euclidean(p1, p2)
print(res)
#Answer:   9.21954445729  
print()

# Is the distance computed using 4 degrees of movement.
# E.g. we can only move: up, down, right, or left, not diagonally.
from scipy.spatial.distance import cityblock
p1 = (1, 0)
p2 = (10, 2)
res = cityblock(p1, p2)
print(res)
#Answer:    11 
print()

# Find the cosine distsance between given points:
from scipy.spatial.distance import cosine
p1 = (1, 0)
p2 = (10, 2)
res = cosine(p1, p2)
print(res)
#Answer:    0.019419324309079777 
print()

from scipy.spatial.distance import hamming
p1 = (True, False, True)
p2 = (False, True, True)
res = hamming(p1, p2)
print(res)
#Answer:     0.666666666667
print()