#!/usr/bin/python3
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def q(conn, query):
    return conn.execute(query)

def sz_vs_perf(conn, x, y, z):
    [(min_perf, max_perf, min_sz, max_sz)] = q(conn, "select min(perf), max(perf), min(sz), max(sz) from b")
    fig, ax = plt.subplots()
    ax.set_ylim(ymin=0, ymax=max_perf)
    ax.set_xlabel("{x}".format(x=x))
    ax.set_ylabel("flops/cycle")
    for i, (yv,) in enumerate(q(conn, "select distinct {y} from b order by {y}".format(y=y))):
        # if i != 0: continue
        for j, (zv,) in enumerate(q(conn, "select distinct {z} from b order by {z}".format(z=z))):
            # if j % 3 != 0: continue
            data = list(q(conn,
                          """select {x},perf from b
                          where {y}={yv} and {z}={zv}
                          order by {x}""".format(x=x, y=y, z=z, yv=yv, zv=zv)))
            X = [x for x, y in data]
            Y = [y for x, y in data]
            ax.plot(X, Y, "*-", label="{y}={yv}, {z}={zv}".format(y=y, yv=yv, z=z, zv=zv))
            ax.legend()
    plt.show()

def intensity_vs_perf(conn):
    data = list(q(conn, "select fma * 1.0 / sz as i, perf from b where i > 34 and perf > 23.5"))
    X = [x for x, y in data]
    Y = [y for x, y in data]
    fig, ax = plt.subplots()
    ax.set_xlabel("fma/size")
    ax.set_ylabel("fmas/cycle")
    ax.scatter(X, Y)
    plt.show()
    
def draw_color(data, x_label, y_label, title,
               min_perf, max_perf, min_sz, max_sz):
    data = list(data)
    Xs = sorted(set([x for x, y, size, flops_per_cycle in data]))
    Ys = sorted(set([y for x, y, size, flops_per_cycle in data]))
    nX = len(Xs)
    nY = len(Ys)
    X = np.zeros((nX, nY))
    Y = np.zeros((nX, nY))
    F = np.zeros((nX, nY))
    for x, y, size, flops_per_cycle in data:
        i = Xs.index(x)
        j = Ys.index(y)
        X[i, j] = x
        Y[i, j] = y
        F[i, j] = flops_per_cycle
    fig, ax = plt.subplots()
    c = ax.pcolor(X, Y, F, vmin=min_perf, vmax=max_perf)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    plt.show()

def grow_two(conn, x, y, z):
    [(min_perf, max_perf, min_sz, max_sz)] = q(conn, "select min(perf), max(perf), min(sz), max(perf) from b")
    for zv, in q(conn, "select distinct {z} from b order by {z}".format(z=z)):
        data = q(conn,
                 """select {x},{y},sz,perf from b
                 where {z}={zv}
                 order by {x},{y}""".format(x=x, y=y, z=z, zv=zv))
        draw_color(data, x, y, "flops/cycle {z} = {zv}".format(z=z, zv=zv),
                   min_perf, max_perf, min_sz, max_sz)
        break
    
def main():
    conn = sqlite3.connect("a.sqlite")
    q(conn, """create temp table b as 
    select M, N, K, avg(repeat * M * N * K * 1.0 / cycles) as perf, 
    M * N * K as fma,
    (M * K + K * N + M * N) * sz_real as sz
    from a group by M, N, K""")
    #grow_two(conn, "M", "N", "K")
    #grow_two(conn, "N", "K", "M")
    #grow_two(conn, "K", "M", "N")
    #sz_vs_perf(conn, "M", "N", "K")
    #sz_vs_perf(conn, "N", "K", "M")
    #sz_vs_perf(conn, "K", "M", "N")
    intensity_vs_perf(conn)
    
main()
