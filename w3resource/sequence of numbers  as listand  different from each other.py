values = input("Input some comma seprated numbers : ")
list = values.split(",")
# set count only uniq number 
def test_distinct(data):
    #length of list== length of uniq number in list
    if len(data) == len(set(data)):
        return True
    else:
        return False
print(test_distinct(list))