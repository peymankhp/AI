def remove_nums(int_list):
      #list starts with 0 index
  position = 3 - 1 
  idx = 0
  len_list = (len(int_list))
  while len_list>0:
    idx = (position+idx)%len_list
    print(int_list.pop(idx))
    len_list -= 1
values = (input("Input some char : "))
nums=values.split()
remove_nums(nums)