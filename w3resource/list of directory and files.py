import os
loc = input('Enter the path you need: ')
print('The files on that path are: ', os.listdir(loc))