i = int(input('put any number and get the digits result: '))
print(eval(' '.join([x for x in str(i)]).replace(' ','+')))
#:O