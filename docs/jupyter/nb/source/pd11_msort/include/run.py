#!/usr/bin/python3
import os

def run1(cmd):
    print(cmd)
    return os.system(cmd)

def run(exes, sizes, threads, repeats=3):
    os.makedirs("out", exist_ok=True)
    for exe in exes:
        for n in sizes:
            for h in threads:
                for t in range(repeats):
                    r = run1("OMP_NUM_THREADS={h} ./{exe} {n} > out/out_{exe}_{h}_{t}.txt"
                             .format(exe=exe, n=n, h=h, t=t))
                    if r:
                        return r
    print("done")
    return 0


if 0:
    run(["msort_2_1.exe", "msort_2_2.exe"],
        [ 10 * 1000 * 1000 ],
        [ 1, 2, 3, 4, 6, 8 ])
    
