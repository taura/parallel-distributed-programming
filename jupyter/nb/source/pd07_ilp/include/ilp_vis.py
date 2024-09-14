#!/usr/bin/python3
import re
# from matplotlib import collections  as mc
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np

def read_dat(files_dat):
    pat = re.compile("thread=(?P<thread>\d+) vcore0=(?P<vcore0>\d+) vcore1=(?P<vcore1>\d+) x=(\d+\.\d+)(,\d+\.\d+)*(?P<t>( \d+)*)")
    log = {}
    for file_dat in files_dat:
        with open(file_dat) as fp:
            for line in fp:
                # 1 : 100.000000 20524414966449 20524423007875 0 0
                m = pat.match(line)
                if not m:
                    continue
                thread = int(m.group("thread"))
                # x      = float(m.group("x"))
                vcore0 = int(m.group("vcore0"))
                vcore1 = int(m.group("vcore1"))
                t      = [int(s) for s in m.group("t").strip().split()]
                # assert(vcore0 == vcore1), (vcore0, vcore1)
                if vcore0 not in log:
                    log[vcore0] = []
                log[vcore0].append((thread, t))
    return log

def ilp_plt(files_dat, start_t=0, end_t=float("inf"), start_thread=0, end_thread=float("inf")):
    log = read_dat(files_dat)
    n_vcores = max(vcore for vcore in log) + 1
    cmap = plt.cm.get_cmap('RdYlGn', n_vcores)
    fig, ax = plt.subplots()
    plt.xlabel("cycles")
    plt.ylabel("thread idx")
    T0 = min(min(T[0] for thread, T in records) for vcore,records in sorted(list(log.items())))
    for vcore,records in sorted(list(log.items())):
        X = []
        Y = []
        vcore_color = cmap(vcore)
        for thread, T in records:
            if start_thread <= thread < end_thread:
                for t in T:
                    if start_t <= t - T0 <= end_t:
                        X.append(t - T0)
                        Y.append(thread)
        ax.plot(X, Y, 'o', markersize=0.5, color=vcore_color)
    ax.autoscale()
    plt.savefig("sched.svg")
    plt.show()
    
