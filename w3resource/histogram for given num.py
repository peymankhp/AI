num = input('number:   ')
res = [int(x) for x in str(num)] 
print(str(res))
for i in res:
    n = i
    print('*'*n, end =" ")