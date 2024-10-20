#!/usr/bin/python3
import re
# from matplotlib import collections  as mc
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np

def read_dat(files_dat):
    pat = re.compile(r"i=(?P<i>\d+) x=(?P<x>\d+\.\d+) thread0=(?P<thread0>\d+) cpu0=(?P<cpu0>\d+) thread1=(?P<thread1>\d+) cpu1=(?P<cpu1>\d+)(?P<t>( \d+)*)")
    log = {}
    for file_dat in files_dat:
        with open(file_dat) as fp:
            for line in fp:
                # 1 : 100.000000 20524414966449 20524423007875 0 0
                m = pat.match(line)
                if not m:
                    continue
                i = int(m.group("i"))
                thread0 = int(m.group("thread0"))
                thread1 = int(m.group("thread1"))
                cpu0 = int(m.group("cpu0"))
                cpu1 = int(m.group("cpu1"))
                x      = float(m.group("x"))
                t      = [int(s) for s in m.group("t").strip().split()]
                assert(thread0 == thread1), (thread0, thread1)
                if thread0 not in log:
                    log[thread0] = []
                log[thread0].append((i, t))
    return log

def sched_plt(files_dat, start_t=0, end_t=float("inf"), start_i=0, end_i=float("inf")):
    log = read_dat(files_dat)
    n_threads = max(thread for thread in log) + 1
    # cmap = plt.cm.get_cmap('RdYlGn', n_threads)
    cmap = plt.get_cmap('RdYlGn', n_threads)
    fig, ax = plt.subplots()
    plt.xlabel("ns")
    plt.ylabel("iteration idx")
    T0 = min(t for records in log.values() for _, T in records for t in T[:1])
    for thread,records in sorted(list(log.items())):
        X = []
        Y = []
        thread_color = cmap(thread)
        for i, T in records:
            if start_i <= i < end_i:
                for t in T:
                    if start_t <= t - T0 <= end_t:
                        X.append(t - T0)
                        Y.append(i)
        ax.plot(X, Y, 'o', markersize=0.5, color=thread_color)
    ax.autoscale()
    plt.savefig("sched.svg")
    plt.show()
    
sched_plt(["a.dat"])

