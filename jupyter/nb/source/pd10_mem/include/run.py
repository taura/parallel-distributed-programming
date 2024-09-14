#!/usr/bin/python3
import os

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

def run1(cmd):
    print(cmd)
    return os.system(cmd)

def run(cmd, sizes, strides, a=10*1000*1000, repeats=3):
    os.makedirs("out", exist_ok=True)
    for n in sizes:
        for s in strides:
            for t in range(repeats):
                r = run1("{cmd} -n {n} -s {s} -a {a} > out/out_{n}_{s}_{t}.txt"
                         .format(cmd=cmd, n=n, s=s, a=a, t=t))
                if r:
                    return r
    print("done")
    return 0


if 0:                           #  e.g.,
    run("./ptr_chase",
        # generate a geometric sequence from 2^10 to 2^28
        seq(2 ** 10, 2 ** 29, 1.3, 64, 1),
        # make stride always 64
        [ 64, 4160 ])
    
