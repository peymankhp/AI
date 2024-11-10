
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)
#Answer:      [42  2  3  4  5]    [1 2 3 4 5] 
print()
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)
#Answer:      [42  2  3  4  5]    [42  2  3  4  5]
print()
#Print the value of the base attribute to check if an array owns it's data or not:
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
y = arr.view()
print(x.base)
print(y.base)
#Answer:      None    [1 2 3 4 5]
print()
