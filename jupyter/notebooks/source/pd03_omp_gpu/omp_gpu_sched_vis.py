#!/usr/bin/python3
import re
# from matplotlib import collections  as mc
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np

def read_dat(files_dat):
    pat = re.compile(r"i=(?P<i>\d+) x=(?P<x>\d+\.\d+) sm0=(?P<sm0>\d+) sm1=(?P<sm1>\d+)(?P<t>( \d+)*)")
    log = {}
    for file_dat in files_dat:
        with open(file_dat) as fp:
            for line in fp:
                # 1 : 100.000000 20524414966449 20524423007875 0 0
                m = pat.match(line)
                if not m:
                    continue
                i = int(m.group("i"))
                sm0 = int(m.group("sm0"))
                sm1 = int(m.group("sm1"))
                x      = float(m.group("x"))
                t      = [int(s) for s in m.group("t").strip().split()]
                assert(sm0 == sm1), (sm0, sm1)
                if sm0 not in log:
                    log[sm0] = []
                log[sm0].append((i, t))
    return log

def sched_plt(files_dat, start_t=0, end_t=float("inf"), start_i=0, end_i=float("inf"), show_every=1):
    log = read_dat(files_dat)
    n_sms = max(sm for sm in log) + 1
    # cmap = plt.cm.get_cmap('RdYlGn', n_threads)
    cmap = plt.get_cmap('RdYlGn', n_sms)
    fig, ax = plt.subplots()
    plt.xlabel("cycles")
    plt.ylabel("iteration idx")
    for sm,records in sorted(list(log.items())):
        T0 = min(t for _, T in records for t in T[:1])
        X = []
        Y = []
        thread_color = cmap(sm)
        for i, T in records:
            if i % show_every == 0 and start_i <= i < end_i:
                for t in T:
                    if start_t <= t - T0 <= end_t:
                        X.append(t - T0)
                        Y.append(i)
        ax.plot(X, Y, 'o', markersize=0.5, color=thread_color)
    ax.autoscale()
    plt.savefig("sched.svg")
    plt.show()
    

