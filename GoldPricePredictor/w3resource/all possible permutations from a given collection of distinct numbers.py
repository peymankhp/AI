from itertools import permutations
values = (input("Input some char : "))
for L in list(permutations(values)): print(*L)
print('Number of possible scenarios: ',2**(len(values))-2)

import itertools
print(list(itertools.permutations(values)))