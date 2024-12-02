import re
import matplotlib.pyplot as plt

# copy the result below
result = """
=== C=1 ===
thread 0 : cycles/iter = ..., fmas/cycle = ...
fmas/cycle = ...
=== C=2 ===

   ...

=== C=12 ===
thread 0 : cycles/iter = ..., fmas/cycle = ...
fmas/cycle = ...
"""

def main():
    C = []
    cycles_per_iter = []
    fmas_per_cycle = []
    for line in result.strip().split("\n"):
        m = re.match("=== C=(?P<C>\d+) ===", line)
        if m:
            c = int(m.group("C"))
        m = re.match("thread (\d+) : cycles/iter = (?P<cpi>\d+\.\d+), fmas/cycle = (?P<fpc>\d+\.\d+)", line)
        if m:
            C.append(c)
            cycles_per_iter.append(float(m.group("cpi")))
            fmas_per_cycle.append(float(m.group("fpc")))
    cycles_per_iter_line, = plt.plot(C, cycles_per_iter, "-*")
    cycles_per_iter_line.set_label("cycles/iter")
    fmas_per_cycle_line, = plt.plot(C, fmas_per_cycle, "-*")
    fmas_per_cycle_line.set_label("fmas/cycle")
    plt.legend()
    plt.show()
    
main()            
