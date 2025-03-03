#No1
print("Input two integers(a b): ")
a,b = map(int,input().split(" "))
print("Number of digit of a and b.:")
print(str(a+b))

#No2
print("Input three integers(sides of a triangle)")
int_num = list(map(int,input().split()))
x,y,z = sorted(int_num)
if x**2+y**2==z**2:
    print('Yes')
else:
    print('No')