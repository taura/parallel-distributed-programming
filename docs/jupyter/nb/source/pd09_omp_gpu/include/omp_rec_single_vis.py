#!/usr/bin/python3

import re
import numpy as np
import matplotlib.pyplot as plt

def main():
    fp = open("rec.txt")
    for line in fp:
        m = re.match("size: (?P<M>\d+)", line)
        if m:
            T = int(m.group("M"))
            N = int(np.sqrt(T))
            M = (T + N - 1) // N
            T = M * N
            C = np.zeros(T)
            C[:] = -1
        m = re.match("x\[(?P<i>\d+)\] = (?P<x>\d+)", line)
        if m:
            [i, x] = [int(x) for x in m.group("i", "x")]
            #print("C[{i},{j}] = {x}".format(i=i, j=j, x=x))
            C[i] = x
    X = np.arange(0, M, 1.0)
    Y = np.arange(0, N, 1.0)
    X, Y = np.meshgrid(Y, X)
    C = C.reshape((M, N))
    fig, ax = plt.subplots()
    c = ax.pcolor(X, Y, C)
    fig.colorbar(c, ax=ax)
    plt.show()
        
main()
