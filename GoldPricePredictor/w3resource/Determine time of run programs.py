import cProfile # ravesh aval 1
import time # ravesh dovom 2
def primenum():
    start_time = time.time() # ravesh dovom 2
    number = 999990
    while number <1000337:
        if number > 1:
            for i in range(2, number):
                if (number % i) == 0:
                    break
            else:
                print(number)
        number +=  1   
    end_time = time.time() # ravesh dovom 2
    time1= end_time-start_time # ravesh dovom 2
    print("import time :     ",time1)  # ravesh dovom 2
cProfile.run('primenum()') # ravesh aval 1