#!/usr/bin/python3
import sys

def is_prime(x):
    """ return if x is prime or not """
    if x == 1:
        return 0
    for d in range(2, x):
        if x % d == 0:
            return 0
        if d * d > x:
            return 1
    return 1
    
def find_prime(a):
    """ find the smallest prime >= a """
    for x in range(a, 100 * a):
        if is_prime(x):
            return x
    assert(0)

def seq(a, b, r, u, prime):
    """
    generate numbers a, ar, ar^2, ar^3, ..., b
    but make each number a multiple of u.
    also, if prime=True, make each number
    a prime multiple of u.
    """
    S = []
    x = a
    while x <= b:
        y = (int(x) + u - 1) // u
        z = find_prime(y) if prime else y
        if z * u <= b:
            S.append(z * u)
        x = x * r
    return sorted(set(S))

def main():
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    r = float(sys.argv[3])
    u = int(sys.argv[4])
    p = int(sys.argv[5])
    for x in seq(a, b, r, u, p):
        print(x)
            
if __name__ == "__main__":
    #seq(400, 128000, 1.3, 64, 1)
    main()
    
