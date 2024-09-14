#!/usr/bin/python3

import re
import numpy as np
import matplotlib.pyplot as plt

def main():
    fp = open("rec.txt")
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
    X = np.arange(0, M, 1.0)
    Y = np.arange(0, N, 1.0)
    X, Y = np.meshgrid(Y, X)
    fig, ax = plt.subplots()
    c = ax.pcolor(X, Y, C)
    fig.colorbar(c, ax=ax)
    plt.show()
        
main()
