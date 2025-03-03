import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
arr = np.array([
  [0, 1, 2],
  [1, 0, 0],
  [2, 0, 0]
])
newarr = csr_matrix(arr)
print(connected_components(newarr))
#Answer:  (1, array([0, 0, 0], dtype=int32))
print()

# Use the dijkstra method to find the shortest path in a graph from one element to another.
# It takes following arguments:
#     return_predecessors: boolean (True to return whole path of traversal otherwise False).
#     indices: index of the element to return all paths from that element only.
#     limit: max weight of path.

#Find the shortest path from element 1 to 2:
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
arr = np.array([
  [0, 1, 2],
  [1, 0, 0],
  [2, 0, 0]
])
newarr = csr_matrix(arr)
print(dijkstra(newarr, return_predecessors=True, indices=0)) 
#Answer:  (array([ 0.,  1.,  2.]), array([-9999,     0,     0], dtype=int32))
print()

#Find the shortest path between all pairs of elements:
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
arr = np.array([
  [0, 1, 2],
  [1, 0, 0],
  [2, 0, 0]
])
newarr = csr_matrix(arr)
print(floyd_warshall(newarr, return_predecessors=True))
#Answer:  (array([[ 0.,  1.,  2.],
    #    [ 1.,  0.,  3.],
    #    [ 2.,  3.,  0.]]), array([[-9999,     0,     0],
    #    [    1, -9999,     0],
    #    [    2,     0, -9999]], dtype=int32))
print()

import numpy as np
from scipy.sparse.csgraph import bellman_ford
from scipy.sparse import csr_matrix
arr = np.array([
  [0, -1, 2],
  [1, 0, 0],
  [2, 0, 0]
])
newarr = csr_matrix(arr)
print(bellman_ford(newarr, return_predecessors=True, indices=0))
#Answer:  (array([ 0., -1.,  2.]), array([-9999,     0,     0], dtype=int32))
print()

import numpy as np
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import csr_matrix
arr = np.array([
  [0, 1, 0, 1],
  [1, 1, 1, 1],
  [2, 1, 1, 0],
  [0, 1, 0, 1]
])
newarr = csr_matrix(arr)
print(depth_first_order(newarr, 1))
#Answer:  (array([1, 0, 3, 2], dtype=int32), array([    1, -9999,     1,     0], dtype=int32))
print()

import numpy as np
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse import csr_matrix
arr = np.array([
  [0, 1, 0, 1],
  [1, 1, 1, 1],
  [2, 1, 1, 0],
  [0, 1, 0, 1]
])
newarr = csr_matrix(arr)
print(breadth_first_order(newarr, 1))
#Answer:  (array([1, 0, 2, 3], dtype=int32), array([    1, -9999,     1,     1], dtype=int32))
print()