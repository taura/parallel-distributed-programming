#!/usr/bin/python3

import re
import numpy as np
import matplotlib.pyplot as plt

def main():
    fp = open("rec.txt")
    max_i = 0
    max_j = 0
    max_x = -1
    for line in fp:
        m = re.match("size: (?P<M>\d+) (?P<N>\d+)", line)
        if m:
            [M, N] = [int(x) for x in m.group("M", "N")]
            C = np.zeros((M, N))
        m = re.match("x\[(?P<i>\d+)\]\[(?P<j>\d+)\] = (?P<x>\d+)", line)
        if m:
            [i, j, x] = [int(x) for x in m.group("i", "j", "x")]
            #print("C[{i},{j}] = {x}".format(i=i, j=j, x=x))
            C[i,j] = x
            if x >= 0:
                max_i = max(max_i, i)
                max_j = max(max_j, j)
                max_x = max(max_x, x)
    M = max_i + 10
    N = max_j + 10
    X = np.arange(0, M, 1.0)
    Y = np.arange(0, N, 1.0)
    X, Y = np.meshgrid(Y, X)
    C = C[:M, :N]
    for i in range(M):
        for j in range(N):
            if C[i,j] < 0:
                C[i,j] = -0.25 * max_x
    fig, ax = plt.subplots()
    c = ax.pcolor(X, Y, C)
    fig.colorbar(c, ax=ax)
    plt.show()
        
main()
