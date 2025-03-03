def divisor(n):
    x=[]
    for i in range(n):
        x=([i for i in range(1,n+1) if not n % i])
    return x
print(divisor(768))