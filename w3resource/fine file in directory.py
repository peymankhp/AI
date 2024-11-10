import os
file = (input('which file you looking for?: '))
if file in os.listdir():
    print('the file is there')
else:
    print('file is not found')