answer = input("For convert inch to cm type 'I' and for cm to inch type 'C': \n")
if answer=='I':
    h_ft = int(input("Feet: "))
    h_inch = int(input("Inches: "))

    h_inch += h_ft * 12
    h_cm = round(h_inch * 2.54, 1)

    print("Your height is : %d cm." % h_cm)
elif answer=='C':
    h_m = int(input("Meter: "))
    h_cm = int(input("cm: "))

    h_cm += h_m * 100
    h_inch = round(h_cm / 2.54, 1)
    h_feet=int(h_inch/12)
    inchaswer=int(h_inch%12)

    print("Your inch is : %d feet" % h_feet , inchaswer ,'inch')
else:
    print('please Enter I or C')