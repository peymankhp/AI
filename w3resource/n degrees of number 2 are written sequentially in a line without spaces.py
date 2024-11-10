num=input('Enter number:   ')
def ndegrees(num):
  ans = True
  n , i , tempn =  2 , 2 , 2
  while ans:
    if str(tempn) in num:
      i += 1
      tempn = pow(n, i)
    else:
      ans = False
  return i-1;
print(ndegrees(num))