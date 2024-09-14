#!/usr/bin/python3
import re
# from matplotlib import collections  as mc
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np

def read_dat(files_dat):
    pat = re.compile("thread=(?P<thread>\d+) x=(?P<x>\d+\.\d+) sm0=(?P<sm0>\d+) sm1=(?P<sm1>\d+)(?P<t>( \d+)*)")
    log = {}
    for file_dat in files_dat:
        with open(file_dat) as fp:
            for line in fp:
                # 1 : 100.000000 20524414966449 20524423007875 0 0
                m = pat.match(line)
                if not m:
                    continue
                thread = int(m.group("thread"))
                x      = float(m.group("x"))
                sm0    = int(m.group("sm0"))
                sm1    = int(m.group("sm1"))
                t      = [int(s) for s in m.group("t").strip().split()]
                assert(sm0 == sm1), (sm0, sm1)
                if sm0 not in log:
                    log[sm0] = []
                log[sm0].append((thread, t))
    return log

def cuda_sched_plt(files_dat, start_t=0, end_t=float("inf"), start_thread=0, end_thread=float("inf")):
    log = read_dat(files_dat)
    n_sms = max(sm for sm in log) + 1
    cmap = plt.cm.get_cmap('RdYlGn', n_sms)
    fig, ax = plt.subplots()
    plt.xlabel("cycles")
    plt.ylabel("thread idx")
    for sm,records in sorted(list(log.items())):
        T0 = min(T[0] for thread, T in records)
        X = []
        Y = []
        sm_color = cmap(sm)
        for thread, T in records:
            if start_thread <= thread < end_thread:
                for t in T:
                    if start_t <= t - T0 <= end_t:
                        X.append(t - T0)
                        Y.append(thread)
        ax.plot(X, Y, 'o', markersize=0.5, color=sm_color)
    ax.autoscale()
    plt.savefig("sched.svg")
    plt.show()
    
cuda_sched_plt(["rec.txt"])

