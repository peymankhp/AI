import math
a=int(input('a:  '))
b=int(input('b:  '))
x=int(input('x:  '))
y=int(input('y:  '))
p1 = [a, b]
p2 = [x, y]
distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
print(distance)
