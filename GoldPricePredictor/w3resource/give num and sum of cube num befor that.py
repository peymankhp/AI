def sum_of_cubes(n):
    n -= 1
    total = 0
    while n > 0:
        total += n * n * n
        n -= 1
    return total
print("Sum of cubes: ",sum_of_cubes(8))