import json

# some JSON:
x =  '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["age"]) 


import json
print(json.dumps({'Alex': 1, 'Suresh': 2, 'Agnessa': 3}))