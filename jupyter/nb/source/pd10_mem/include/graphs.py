#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def q(conn, query):
    print(query)
    return conn.execute(query)

def size_vs_lat_seq(conn):
    strides = list(q(conn, "select distinct stride from a order by stride"))
    fig, ax = plt.subplots()
    for stride, in strides[:4]:
        result = list(q(conn,
                        ("""
                        select size, avg(cycles * 1.0 / accesses) from a 
                        where stride = {stride} 
                        group by size
                        order by size
                        """
                         .format(stride=stride))))
        X = [size for size, lat in result]
        Y = [lat for size, lat in result]
        line, = ax.plot(X, Y, "*-", label="stride={stride}".format(stride=stride))
    ax.set_xlabel("size")
    ax.set_ylabel("latency")
    plt.legend()
    plt.xscale("log")
    plt.ylim(0)
    plt.show()

def main():
    conn = sqlite3.connect("a.sqlite")
    size_vs_lat_seq(conn)

if __name__ == "__main__":
    main()
    

        

