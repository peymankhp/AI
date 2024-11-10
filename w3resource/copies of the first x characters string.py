string = input('enter a sentence>>>')
n = int(input('enter number of copies>>>'))
mn = int(input('enter number of select sentence>>>'))
def function(string,n):
    length = len(string)
    result = ''
    if length < mn:
        for i in range(n):
            result = result + string
        return result
    else:
        for i in range(n):
            result = result + string[:mn]
        return result
print(function(string,n))