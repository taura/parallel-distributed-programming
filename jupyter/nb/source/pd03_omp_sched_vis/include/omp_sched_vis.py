#!/usr/bin/python3

import colorsys
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython import display 

def read_log(log_txt):
    fp = open(log_txt)
    p = re.compile("(?P<r>\d+) (?P<i>\d+) (?P<j>\d+)"
                   " (?P<begin>\d+) (?P<end>\d+)"
                   " (?P<load>\d+) (?P<thread>\d+)")
    R = {}                      # i,j -> begin,end,load,thread
    for line in fp:
        m = p.match(line)
        assert m, line
        [ r,i,j,begin,end,load,thread ] = [ int(f) for f in m.groups() ]
        R[r,i,j] = (begin,end,load,thread)
    fp.close()
    return R

def make_norm_rgb_table(threads):
    tbl = {}
    n = len(threads)
    for i,w in enumerate(threads):
        h = float(i)/float(n)
        r,g,b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        tbl[w] = r,g,b
    return tbl

def sched_vis(log_txt):
    """
    R : (i,j) -> begin,end,load,thread
    """
    R = read_log(log_txt)
    # get the set of unique thread ids
    threads = set([ t for _,_,_,t in R.values() ])
    max_load = max([ l for _,_,l,_ in R.values() ])
    # assign color to each thread
    rgb_table = make_norm_rgb_table(threads)
    # get number of rows and columns
    repeat = max([ r for r,i,j in R.keys() ]) + 1
    M = max([ i for r,i,j in R.keys() ]) + 1
    N = max([ j for r,i,j in R.keys() ]) + 1
    img = np.zeros((M,N,3))
    fig = plt.figure(figsize=(M / 72, N / 72))
    im = plt.imshow(img)
    interval_ms = 30
    speed_factor = 0.2
    draw_interval_clock = interval_ms * 1.0e6 * speed_factor
    def sort_key(rec):
        (r,i,j),(begin,end,load,thread) = rec
        return end
    
    def load_data():
        events = sorted(R.items(), key=sort_key)
        max_r = -1
        for c,((r,i,j),(begin,end,load,thread)) in enumerate(events):
            if c == 0 or end - last_update > draw_interval_clock:
                last_update = end
                im.set_data(img)
                yield im,
            rgb = rgb_table[thread]
            #alpha = 1.0 # if load/float(max_load) > 0.005 else 0.4
            assert (r >= max_r)
            if r > max_r:
                img[:,:,:] = np.zeros((M,N,3))
                max_r = r
            img[i,j] = rgb
        im.set_data(img)
        yield im,

    iterator = load_data()
    def update(*args):
        try:
            return next(iterator)
        except StopIteration:
            return im,
    ani = animation.FuncAnimation(fig, update, interval=interval_ms, blit=True)
    html = display.HTML(ani.to_jshtml())
    display.display(html)
    plt.close()
    # plt.show()

# usage:
# sched_vis("log.txt")
