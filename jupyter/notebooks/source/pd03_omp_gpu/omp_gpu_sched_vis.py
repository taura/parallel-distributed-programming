#!/usr/bin/python3
import re
# from matplotlib import collections  as mc
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np

def read_dat(files_dat):
    pat = re.compile(r"i=(?P<i>\d+) x=(?P<x>\d+\.\d+) team0=(?P<team0>\d+) thread0=(?P<thread0>\d+) sm0=(?P<sm0>\d+) team1=(?P<team1>\d+) thread1=(?P<thread1>\d+) sm1=(?P<sm1>\d+)(?P<t>( \d+)*)")
    log = {}
    for file_dat in files_dat:
        with open(file_dat) as fp:
            for line in fp:
                # 1 : 100.000000 20524414966449 20524423007875 0 0
                m = pat.match(line)
                if not m:
                    continue
                i = int(m.group("i"))
                team0 = int(m.group("team0"))
                team1 = int(m.group("team1"))
                thread0 = int(m.group("thread0"))
                thread1 = int(m.group("thread1"))
                sm0 = int(m.group("sm0"))
                sm1 = int(m.group("sm1"))
                x      = float(m.group("x"))
                t      = [int(s) for s in m.group("t").strip().split()]
                assert(team0 == team1), (team0, team1)
                assert(thread0 == thread1), (thread0, thread1)
                assert(sm0 == sm1), (sm0, sm1)
                if (team0, thread0) not in log:
                    log[team0, thread0] = []
                log[team0, thread0].append((i, t))
    return log

def sched_plt(files_dat, start_t=0, end_t=float("inf"), start_i=0, end_i=float("inf")):
    log = read_dat(files_dat)
    n_teams = max(team for team, thread in log) + 1
    n_threads = max(thread for team, thread in log) + 1
    # cmap = plt.cm.get_cmap('RdYlGn', n_threads)
    cmap = plt.get_cmap('RdYlGn', n_teams * n_threads)
    fig, ax = plt.subplots()
    plt.xlabel("ns")
    plt.ylabel("iteration idx")
    T0 = min(T[0] for records in log.values() for _, T in records)
    for (team, thread),records in sorted(list(log.items())):
        # T0 = min(T[0] for thread, T in records)
        X = []
        Y = []
        thread_color = cmap(team * n_threads + thread)
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
    

