# The savemat() function allows us to export data in Matlab format.
# The method takes the following parameters:
#     filename - the file name for saving data.
#     mdict - a dictionary containing the data.
#     do_compression - a boolean value that specifies wheter to compress the reult or not. Default False.

from scipy import io
import numpy as np
arr = np.arange(10)
io.savemat('arr.mat', {"vec": arr})

# Import Data from Matlab Format
# The loadmat() function allows us to import data from a Matlab file.
# The function takes one required parameter:
# filename - the file name of the saved data.
# It will return a structured array whose keys are the variable names, and the corresponding values are the variable values.

from scipy import io
import numpy as np
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,])
# Export:
io.savemat('arr.mat', {"vec": arr})
# Import:
mydata = io.loadmat('arr.mat')
print(mydata) 
#Answer:    {
#   '__header__': b'MATLAB 5.0 MAT-file Platform: nt, Created on: Tue Sep 22 13:12:32 2020',
#   '__version__': '1.0',
#   '__globals__': [],
#   'vec': array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
# }
print()

from scipy import io
import numpy as np
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,])
#Export:
io.savemat('arr.mat', {"vec": arr})
#Import:
mydata = io.loadmat('arr.mat')
print(mydata['vec'])
#Answer:     [[0 1 2 3 4 5 6 7 8 9]] 
print()

from scipy import io
import numpy as np
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,])
#Export:
io.savemat('arr.mat', {"vec": arr})
#Import:
mydata = io.loadmat('arr.mat', squeeze_me=True)
print(mydata['vec'])
#Answer:     [0 1 2 3 4 5 6 7 8 9] 
print()