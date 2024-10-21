#!/usr/bin/python3

import colorsys,re,sys,optparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def gen_lb_gpl(R, opts):
    # R : (i,j) -> begin,end,load,thread
    M = max([ i for i,j in R.keys() ]) + 1
    N = max([ j for i,j in R.keys() ]) + 1
    threads = set([ t for _,_,_,t in R.values() ])
    thread_idx = {}
    for i,t in enumerate(threads):
        thread_idx[t] = i
    wp = open(opts.lb_gpl, "wb")
    wp.write("set view map\n")
    wp.write("set size square\n")
    wp.write("splot '-' matrix with image\n")
    for i in range(M):
        for j in range(N):
            begin,end,load,thread = R[i,j]
            #wp.write("%s %s %s\n" % (i, j, thread_idx[thread]))
            wp.write("%s " % thread_idx[thread])
        wp.write("\n")
    wp.write("e\n")
    wp.write("e\n")
    wp.close()

#
# load balancing map 
#

def make_rgb_table(threads, depth):
    tbl = {}
    n = len(threads)
    for i,w in enumerate(threads):
        h = float(i)/float(n)
        r,g,b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        tbl[w] = (int(r * depth), int(g * depth), int(b * depth))
    return tbl

def make_norm_rgb_table(threads):
    tbl = {}
    n = len(threads)
    for i,w in enumerate(threads):
        h = float(i)/float(n)
        r,g,b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        tbl[w] = r,g,b
    return tbl

def gen_lb_ppm(R, opts):
    """
    R : (i,j) -> begin,end,load,thread
    """
    # get the set of unique thread ids
    threads = set([ t for _,_,_,t in R.values() ])
    # assign color to each thread
    rgb_table = make_rgb_table(threads, opts.color_depth)
    # get number of rows and columns
    M = max([ i for r,i,j in R.keys() ]) + 1
    N = max([ j for r,i,j in R.keys() ]) + 1
    wp = open(opts.lb_ppm, "wb")
    wp.write("P3\n")
    wp.write("%s %s\n" % (M, N))
    wp.write("255\n")
    for i in range(M):
        for j in range(N):
            begin,end,load,thread = R[0,i,j]
            rgb = rgb_table[thread]
            wp.write("%s %s %s\n" % rgb)
    wp.close()

def gen_lb_static_plt(R, opts):
    """
    R : (i,j) -> begin,end,load,thread
    """
    # get the set of unique thread ids
    threads = set([ t for _,_,_,t in R.values() ])
    # assign color to each thread
    rgb_table = make_norm_rgb_table(threads)
    # get number of rows and columns
    M = max([ i for i,j in R.keys() ]) + 1
    N = max([ j for i,j in R.keys() ]) + 1
    img = np.zeros((M, N, 3))
    fig = plt.figure(figsize=(M/72, N/72))
    for i in range(M):
        for j in range(N):
            begin,end,load,thread = R[i,j]
            rgb = rgb_table[thread]
            img[i,j] = rgb_table[thread]
    im = plt.imshow(img)
    plt.show()

def gen_lb_dyn_plt(R, opts):
    """
    R : (i,j) -> begin,end,load,thread
    """
    # get the set of unique thread ids
    threads = set([ t for _,_,_,t in R.values() ])
    max_load = max([ l for _,_,l,_ in R.values() ])
    # assign color to each thread
    rgb_table = make_norm_rgb_table(threads)
    # get number of rows and columns
    repeat_ = max([ r for r,i,j in R.keys() ]) + 1
    M_ = max([ i for r,i,j in R.keys() ]) + 1
    N_ = max([ j for r,i,j in R.keys() ]) + 1
    A = 32
    M = (M_ + A - 1) // A
    N = (N_ + A - 1) // A
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
            img[i//A,j//A] = rgb
        im.set_data(img)
        yield im,

    iterator = load_data()
    def update(*args):
        try:
            return next(iterator)
        except StopIteration:
            return im,

    ani = animation.FuncAnimation(fig, update, interval=interval_ms, blit=True)
    plt.show()

def gen_ts_gpl(R, opts):
    threads = set([ t for _,_,_,t in R.values() ])
    # determine the range of time
    min_begin = min([ begin for begin,_,_,_ in R.values() ])
    max_end   = max([ end   for _,end,_,_   in R.values() ])
    assert (min_begin < max_end), (min_begin, max_end)
    # and a list of (thread_idx, thread, begin, end, load)
    intervals = {}
    for thread in threads:
        intervals[thread] = []
    for begin,end,load,thread in R.values():
        intervals[thread].append((begin - min_begin,end - min_begin,load))
    wp = open(opts.ts_gpl, "wb")
    for i,(thread,ints) in enumerate(intervals.items()):
        ints.sort()
        t_last = 0
        for a,b,load in ints:
            for j in xrange(t_last, a):
                wp.write("%s %s %s\n" % (j, i, -1))
            for j in xrange(a, b):
                wp.write("%s %s %s\n" % (j, i, i))
            t_last = b
        for j in xrange(t_last, opts.width):
            wp.write("%s %s %s\n" % (j, i, -1))
    wp.close()

#
# time_series graph in ppm
#
def gen_ts_ppm(R, opts):
    threads = set([ t for _,_,_,t in R.values() ])
    rgb_table = make_rgb_table(threads, opts.color_depth)
    white_rgb = (opts.color_depth, opts.color_depth, opts.color_depth)
    # determine the range of time
    min_begin = min([ begin for begin,_,_,_ in R.values() ])
    max_end   = max([ end   for _,end,_,_   in R.values() ])
    assert (min_begin < max_end), (min_begin, max_end)
    # and a list of (thread_idx, thread, begin, end, load)
    intervals = {}
    for thread in threads:
        intervals[thread] = []
    for begin,end,load,thread in R.values():
        intervals[thread].append((begin - min_begin,end - min_begin,load))
    span = max_end - min_begin
    n_threads = len(threads)
    wp = open(opts.ts_ppm, "wb")
    wp.write("P3\n")
    wp.write("%s %s\n" % (opts.width, opts.height))
    wp.write("255\n")
    for i,(thread,ints) in enumerate(intervals.items()):
        ints.sort()
        rgb = rgb_table[thread]
        i0 = ( i      * opts.height) / n_threads
        i1 = ((i + 1) * opts.height) / n_threads
        for i in range(i0,i1):
            # draw horizontal lines
            t_last = 0
            for a,b,load in ints:
                for j in xrange((t_last * opts.width) / span, (a * opts.width) / span):
                    wp.write("# %s %s\n" % (i, j))
                    wp.write("%s %s %s\n" % white_rgb)
                for j in xrange((a * opts.width) / span, (b * opts.width) / span):
                    wp.write("# %s %s\n" % (i, j))
                    wp.write("%s %s %s\n" % rgb)
                t_last = b
            for j in xrange((t_last * opts.width) / span, opts.width):
                wp.write("# %s %s\n" % (i, j))
                wp.write("%s %s %s\n" % white_rgb)
    wp.close()

def read_logx(fp):
    p = re.compile(r"(?P<r>\d+) (?P<i>\d+) (?P<j>\d+)"
                   r" (?P<begin>\d+) (?P<end>\d+)"
                   r" (?P<load>\d+) (?P<thread>\d+)")
    R = {}                      # i,j -> begin,end,load,thread
    for line in fp:
        m = p.match(line)
        assert m, line
        [ r,i,j,begin,end,load,thread ] = [ int(f) for f in m.groups() ]
        R[r,i,j] = (begin,end,load,thread)
    return R

def read_log(fp):
    p = re.compile(r"i = (?P<i>\d+) j = (?P<j>\d+) t = (?P<begin>\d+) thread = (?P<thread>\d+)")
    R = {}                      # i,j -> begin,end,load,thread
    for line in fp:
        m = p.match(line)
        assert m, line
        [ i,j,begin,thread ] = [ int(f) for f in m.groups() ]
        (r, end, load) = (0, begin + 1, 1)
        R[r,i,j] = (begin,end,load,thread)
    return R

def parse_opts(argv):
    usage = "usage: %prog [options] log_file"
    p = optparse.OptionParser(usage=usage)
    p.add_option("-l", "--lb_gpl", dest="lb_gpl",
                 default="",
                 help="write load balancing gpl to FILE",
                 metavar="FILE")
    p.add_option("--lb_ppm", dest="lb_ppm",
                 default="",
                 help="write load balancing ppm to FILE",
                 metavar="FILE")
    p.add_option("--lb_plt", dest="lb_plt",
                 default=1,
                 help="plot load balancing with matplotlib",
                 metavar="FILE")
    p.add_option("-t", "--ts_gpl", dest="ts_gpl",
                 default="",
                 help="write time series gpl to FILE",
                 metavar="FILE")
    p.add_option("--ts_ppm", dest="ts_ppm",
                 default="",
                 help="write time series ppm to FILE",
                 metavar="FILE")
    p.add_option("--color_depth", dest="color_depth",
                 default=255,
                 help="set color depth to D",
                 metavar="D")
    p.add_option("--width", dest="width",
                 default=800,
                 help="set width to W",
                 metavar="W")
    p.add_option("--height", dest="height",
                 default=128,
                 help="set height to H",
                 metavar="H")
    options,args = p.parse_args(argv[1:])
    if len(args) == 0:
        args = [ "sched.dat" ]
    if len(args) != 1:
        p.print_help()
        return None,None
    return options,args

def main(argv):
    opts,args = parse_opts(argv)
    if opts is None: return 1
    if len(args) == 0:
        log_txt = "sched.dat"
    else:
        [ log_txt ] = args
    fp = open(log_txt)
    R = read_log(fp)
    fp.close()
    if 1:
        gen_lb_dyn_plt(R, opts)
    else:
        if opts.lb_gpl: 
            gen_lb_gpl(R, opts)
        if opts.lb_ppm: 
            gen_lb_ppm(R, opts)
        if opts.lb_plt: 
            gen_lb_dyn_plt(R, opts)
        if opts.ts_gpl: 
            gen_ts_gpl(R, opts)
        if opts.ts_ppm: 
            gen_ts_ppm(R, opts)
    return 0

if __name__ == "__main__":
    main(sys.argv)


