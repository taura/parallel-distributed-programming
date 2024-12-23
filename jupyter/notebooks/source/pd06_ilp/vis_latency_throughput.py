import matplotlib.pyplot as plt
import re

# ------------------------------------------
# replace the following with your experimental results below
# ------------------------------------------

DATA_STRING = r"""
L=1 M=100 N=10000000 C=1 cycles_per_iter=2.128242
L=1 M=100 N=10000000 C=2 cycles_per_iter=2.126931
L=1 M=100 N=10000000 C=3 cycles_per_iter=2.125129
    ...
"""

def parse(data_s):
    lines = data_s.strip().split("\n")
    p = re.compile(r"L=(?P<L>\d+) M=(?P<M>\d+) N=(?P<N>\d+) C=(?P<C>\d+) cycles_per_iter=(?P<cpi>\d+\.\d+)")
    data = {}
    for line in lines:
        m = p.match(line)
        assert(m), line
        [L, M, N, C, cpi] = [float(x) for x in m.group("L", "M", "N", "C", "cpi")]
        data[C] = {"L" : L, "M" : M, "N" : N, "C" : C, "cpi" : cpi}
    return data

def vis(data_string):
    data = parse(data_string)
    C = sorted(list(data.keys()))
    lat = [data[c]["cpi"] for c in C]
    thr = [data[c]["L"] * c / data[c]["cpi"] for c in C]
    fig, ax0 = plt.subplots()
    #print(lat)
    #print(thr)
    ax0.plot(C, lat, 'b-', label="latency")
    ax0.set_xlabel("C")
    ax0.set_ylabel("latency (cycles/iter)", color='b')
    ax0.set_ylim(bottom=0)
    ax1 = ax0.twinx()
    ax1.plot(C, thr, 'r--', label="throughput")
    ax1.set_xlabel("C")
    ax1.set_ylabel("throughput (FMAs/cycle)", color='r')
    ax1.set_ylim(bottom=0)
    plt.show()
    
