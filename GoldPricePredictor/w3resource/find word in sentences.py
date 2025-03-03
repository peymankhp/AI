import re
x=input("Enter a word:")
z=input("Enter a word:")
if re.findall(x,z):
    print("contain ")
else:
    print("Does not contain")