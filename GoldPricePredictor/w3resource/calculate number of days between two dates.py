from datetime import date
a=int(input('input first year'))
b=int(input('input first month'))
c=int(input('input first day'))
d=int(input('input second year'))
e=int(input('input second month'))
f=int(input('input second day'))
f_date = date(a, b, c)
l_date = date(d, e, f)
delta = l_date - f_date
print(delta.days)