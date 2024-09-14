#!/usr/bin/python3
import glob
import matplotlib.pyplot as plt
import numpy as np

#from read_data import read_files

def size_vs_what(filenames, what="cycles", strides=None):
    # read all data
    dfa = read_files(filenames)
    fig, ax = plt.subplots()
    if strides is None:
        strides = dfa["stride"].unique()
    for stride in strides:
        # get data of the specified stride
        df = dfa[dfa["stride"] == stride]
        avg = df.groupby(["size"]).mean()
        avg = avg.sort_values("size")
        size = np.array(avg.index)
        vals = np.array(avg[what]) / np.array(avg["accesses"])
        label = "stride={stride}".format(stride=stride)
        line, = ax.plot(size, vals, "*-", label=label)
    ax.set_xlabel("size")
    ax.set_ylabel("{what} per access".format(what=what))
    plt.legend()
    plt.xscale("log")
    plt.ylim(0)
    plt.show()

#size_vs_what(glob.glob("out/out_*.txt"), "cycles")
#size_vs_what(glob.glob("out/out_*.txt"), "L1-dcache-load-misses")
#size_vs_what(glob.glob("out/out_*.txt"), "cache-misses")

#size_vs_what(glob.glob("out/out_*.txt"), "l2_lines_in.all")
# size_vs_what(glob.glob("out/out_*.txt"), "offcore_requests.l3_miss_demand_data_rd")

