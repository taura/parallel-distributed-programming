#!/usr/bin/python3
import glob
import matplotlib.pyplot as plt
import numpy as np

#from read_data import read_files

def speedup(filenames, exes=None):
    # read all data
    dfa = read_files(filenames)
    fig, ax = plt.subplots()
    if exes is None:
        exes = dfa["exe"].unique()
    exe0 = exes[0]
    base = dfa[(dfa["exe"] == exe0) & (dfa["threads"] == 1)]["ref_cycles"].agg("mean")
    for exe in exes:
        df = dfa[dfa["exe"] == exe]
        avg = df.groupby(["threads"]).mean()
        avg = avg.sort_values("threads")
        threads = np.array(avg.index)
        perf = base / np.array(avg["ref_cycles"])
        label = exe.strip()
        line, = ax.plot(threads, perf, "*-", label=label)
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup")
    plt.legend()
    plt.xlim(0)
    plt.ylim(0)
    plt.show()

if 0:
    speedup(glob.glob("versioned/out/out_*.txt"))
    
