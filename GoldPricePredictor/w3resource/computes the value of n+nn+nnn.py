n=int(input('type number:  '))
n1=int("%s" % n )
n2=int("%s%s" % (n,n) )
n3=int("%s%s%s" % (n,n,n) )
print(n1+n2+n3)

print(n + int("%i"%n * 2) + int("%i"%n * 3))

n1 = n
n2 = int(str(n)+ str(n))
n3 = int(str(n)+ str(n)+ str(n))
print(n1 + n2 + n3)

print(n*(123))