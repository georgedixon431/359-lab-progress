import numba
import time

def sumSquares (n):
    total = 0
    for i in range (n):
        total += i * i
    return total

@numba.njit
def sumSquaresNumba (n):
    total = 0
    for i in range (n):
        total += i * i
    return total

n = 100000000

totalStart = time.time ()
result = sumSquaresNumba (n)
print("Result:", result)
print("First call:", time.time () - totalStart, "s")

start = time.time ()
result = sumSquaresNumba (n)
print("Result:", result)
print("Second call:", time.time () - start, "s")
print ("Total time:", time.time () - totalStart, "s")

start = time.time ()
sumSquares (n)
print("Python:", time.time() - start)