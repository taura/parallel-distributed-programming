#!/usr/bin/python3
"""Visualize when each loop iteration ran on the GPU, and by which thread.

The data come from gpu_sched_rec.cu / gpu_sched_rec.cc, which print one line
per iteration:

  i=.. sm=.. block=.. thread=.. warp=.. lane=.. gt0=.. gt1=.. x=.. T=t0 t1 ..

sched_plt() draws time on the horizontal axis and the iteration number on the
vertical axis, and colors each point by whichever of sm/block/warp/thread/lane
you ask for.
"""
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

PAT = re.compile(
    r"i=(?P<i>\d+) sm=(?P<sm>\d+)/(?P<nsm>\d+) block=(?P<block>\d+)"
    r" thread=(?P<thread>\d+)"
    r" warp=(?P<hwwarp>\d+) lane=(?P<lane>\d+)"
    r" gt0=(?P<gt0>\d+) gt1=(?P<gt1>\d+) c0=(?P<c0>\d+) c1=(?P<c1>\d+)"
    r" x=(?P<x>[-\d.naif]+) T=(?P<T>[\d ]*)")
PARAM = re.compile(r"#params L=(?P<L>\d+) M=(?P<M>\d+) N=(?P<N>\d+)")

KEYS = ("sm", "block", "warp", "thread", "lane", "gthread", "hwwarp")

class Records(list):
    """a list of records that also carries the run's parameters"""
    params = {"L": 0, "M": 0, "N": 0}
    cycles_per_ns = float("nan")

def read_dat(files):
    """read the files into a list of records (dicts).

    recs.params is {"L":.., "M":.., "N":..} and recs.cycles_per_ns is the
    GPU clock rate measured from the data itself: every iteration read
    %clock64 and %globaltimer at the same two points, so the ratio of the
    two differences is the number of cycles the GPU ticked per nanosecond.
    """
    if isinstance(files, str):
        files = [files]
    recs = Records()
    for f in files:
        with open(f) as fp:
            for line in fp:
                p = PARAM.match(line)
                if p:
                    recs.params = {k: int(p.group(k))
                                   for k in ("L", "M", "N")}
                    continue
                m = PAT.match(line)
                if not m:
                    continue
                r = {k: int(m.group(k))
                     for k in ("i", "sm", "nsm", "block", "thread",
                               "hwwarp", "lane", "gt0", "gt1", "c0", "c1")}
                r["T"] = [int(s) for s in m.group("T").split()]
                recs.append(r)
    if not recs:
        raise Exception("no records found in %s" % (files,))
    # a thread id that is unique across the whole device
    n_threads = max(r["thread"] for r in recs) + 1
    for r in recs:
        r["gthread"] = r["block"] * n_threads + r["thread"]
        # the warp a thread belongs to.  Note this is NOT the %warpid the
        # hardware reports (kept as "hwwarp"): that is the physical warp
        # slot the block was given, which says nothing about which threads
        # move together.  Threads 0..31 are one warp, 32..63 the next, ...
        r["warp"] = r["thread"] // 32
    rates = sorted((r["c1"] - r["c0"]) / float(r["gt1"] - r["gt0"])
                   for r in recs if r["gt1"] > r["gt0"])
    recs.cycles_per_ns = rates[len(rates) // 2] if rates else float("nan")
    return recs

def _colors(recs, color_by):
    """map each record's key to a color.

    With few distinct values (e.g. coloring by SM or by warp) we want colors
    that are easy to tell apart, so we use a qualitative colormap and a
    legend.  With many distinct values (e.g. coloring by thread, of which
    there may be a thousand) no set of colors is distinguishable one by one,
    so we use a continuous colormap and a colorbar instead: you can no longer
    name a thread by its color, but you can still see order and grouping.
    """
    vals = sorted(set(r[color_by] for r in recs))
    if len(vals) <= 20:
        cmap = plt.get_cmap("tab20" if len(vals) > 10 else "tab10")
        table = {v: cmap(k % cmap.N) for k, v in enumerate(vals)}
        return table, None
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=vals[0], vmax=vals[-1])
    return {v: cmap(norm(v)) for v in vals}, cm.ScalarMappable(norm, cmap)

def sched_plt(files, color_by="warp", clock="cycle", align="sm", skip=0,
              start_t=0, end_t=float("inf"),
              start_i=0, end_i=float("inf"), show_every=1,
              markersize=1.0, save="sched.svg", figsize=(9, 6)):
    """draw the schedule.

    color_by : "sm", "block", "warp", "thread", "lane" or "gthread"
    clock    : "cycle"  -> use %clock64, one point per recorded sample.
                           Precise, but only meaningful within a single SM.
               "global" -> use %globaltimer, one horizontal bar per iteration
                           from its start to its end.  Comparable across SMs,
                           so this is the one to use once you have more than
                           one thread block.  It is still shown in GPU cycles,
                           converted with the clock rate measured from the
                           data itself.
    align    : "sm" subtracts each SM's own first clock, "global" subtracts
               a single origin.  Ignored when clock="global".
    skip     : drop this many samples from the start of every iteration.
               The default is 0, i.e. nothing is hidden.
    """
    if color_by not in KEYS:
        raise Exception("color_by must be one of %s" % (KEYS,))
    all_recs = read_dat(files)
    recs = [r for r in all_recs
            if start_i <= r["i"] < end_i and r["i"] % show_every == 0]
    table, mappable = _colors(recs, color_by)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_ylabel("iteration (i)")

    if clock in ("global", "ns"):
        rate = all_recs.cycles_per_ns
        ax.set_xlabel("cycles from the start (%%globaltimer x %.3f GHz)" % rate)
        t0 = min(r["gt0"] for r in recs)
        segs, cols = [], []
        for r in recs:
            a, b = (r["gt0"] - t0) * rate, (r["gt1"] - t0) * rate
            if start_t <= a <= end_t:
                segs.append([(a, r["i"]), (b, r["i"])])
                cols.append(table[r[color_by]])
        ax.add_collection(LineCollection(segs, colors=cols,
                                         linewidths=max(markersize, 1.0)))
    elif clock == "cycle":
        ax.set_xlabel("cycles from the start (%clock64)")
        if align == "sm":
            origin = {}
            for r in recs:
                if len(r["T"]) > skip:
                    origin[r["sm"]] = min(origin.get(r["sm"], r["T"][skip]),
                                          r["T"][skip])
        else:
            o = min(r["T"][skip] for r in recs if len(r["T"]) > skip)
            origin = {r["sm"]: o for r in recs}
        by_color = {}
        for r in recs:
            c = table[r[color_by]]
            X, Y = by_color.setdefault(c, ([], []))
            for t in r["T"][skip:]:
                dt = t - origin[r["sm"]]
                if start_t <= dt <= end_t:
                    X.append(dt)
                    Y.append(r["i"])
        for c, (X, Y) in by_color.items():
            ax.plot(X, Y, "o", markersize=markersize, color=c)
    else:
        raise Exception('clock must be "cycle" or "global"')

    if mappable is None:
        vals = sorted(table)
        if len(vals) <= 12:
            for v in vals:
                ax.plot([], [], "o", color=table[v], label="%s %d"
                        % (color_by, v))
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      fontsize="small")
    else:
        fig.colorbar(mappable, ax=ax, label=color_by)
    ax.autoscale()
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()

def summary(files, skip=0):
    """print a few numbers that are easier to read than the picture.

    Everything is in GPU cycles.  The headline is the last line: how many
    cycles one  x = a * x + b  takes, obtained from the gap between two
    consecutive %clock64 samples divided by N, the number of multiply-adds
    between them.
    """
    recs = read_dat(files)
    rate = recs.cycles_per_ns
    N = recs.params["N"]
    span = (max(r["gt1"] for r in recs) - min(r["gt0"] for r in recs)) * rate
    dur = sorted((r["gt1"] - r["gt0"]) * rate for r in recs)
    gaps = sorted(r["T"][j + 1] - r["T"][j]
                  for r in recs for j in range(skip, len(r["T"]) - 1))
    print("iterations         : %d" % len(recs))
    print("SMs used           : %d of the %d this GPU has"
          % (len(set(r["sm"] for r in recs)), recs[0]["nsm"]))
    print("thread blocks      : %d" % len(set(r["block"] for r in recs)))
    print("warps per block    : %d" % (max(r["warp"] for r in recs) + 1))
    print("threads per block  : %d" % (max(r["thread"] for r in recs) + 1))
    print("GPU clock          : %.3f GHz (measured from the data)" % rate)
    print("wall time          : %.0f cycles" % span)
    print("per iteration      : min %.0f / median %.0f / max %.0f cycles"
          % (dur[0], dur[len(dur) // 2], dur[-1]))
    M = recs.params["M"]
    print("throughput         : %.2f FMA / cycle"
          % (len(recs) * M * N / span if span else float("nan")))
    if gaps and N:
        g = gaps[len(gaps) // 2]
        print("one x = a * x + b  : %.2f cycles"
              "   (%d cycles between samples / N = %d)" % (g / float(N), g, N))

def throughput(f):
    """(threads, FMA per cycle, SMs used by the run, SMs the device has)

    The last one comes from the GPU's own %nsmid register, not from counting
    which SM ids happened to show up, so it is right even for a run that was
    too small to land on every SM.
    """
    recs = read_dat(f)
    M, N = recs.params["M"], recs.params["N"]
    span = ((max(r["gt1"] for r in recs) - min(r["gt0"] for r in recs))
            * recs.cycles_per_ns)
    return (len(recs),
            (len(recs) * M * N / span if span else float("nan")),
            len(set(r["sm"] for r in recs)),   # SMs this run landed on
            recs[0]["nsm"])                    # SMs the device has (%nsmid)

def throughput_plt(series, peak_per_sm=64, peak_sms=None,
                   save="throughput.svg", figsize=(9, 6), log=False):
    """How fast does the GPU go as you give it more threads?

    series : {label: [file, file, ...]}, or just a list of files.
             Every file contributes one point: the number of threads it ran
             (T x H) on the horizontal axis, and the FMA / cycle it reached
             on the vertical one.
    peak_per_sm : the hardware's limit for one SM (64 double-precision
             fused multiply-adds per cycle on this GPU).
    peak_sms : how many SMs the dashed line should assume.  By default it
             is the number the GPU itself reports in %nsmid, which is what
             the biggest run could have used.  Pass peak_sms=1 when every
             run in the plot is confined to a single thread block, since
             then one SM is all it can ever use.
    log   : both axes are linear by default, so "twice the threads, twice
            the speed" is literally a straight line through the origin and
            the flat part where it stops paying is just as plain.  Set
            log=True to spread the small end out instead.
    """
    if not isinstance(series, dict):
        series = {"": list(series)}
    fig, ax = plt.subplots(figsize=figsize)
    device_sms, used_sms = set(), 1
    for label, files in series.items():
        pts = sorted(throughput(f) for f in files)
        device_sms |= set(p[3] for p in pts)
        used_sms = max([used_sms] + [p[2] for p in pts])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                label=label or None)
    if len(device_sms) > 1:
        raise Exception("these files come from GPUs with different SM counts: %s"
                        % sorted(device_sms))
    n_sm = peak_sms if peak_sms else device_sms.pop()
    if peak_per_sm:
        ax.axhline(n_sm * peak_per_sm, ls="--", color="gray")
        ax.text(0.01, n_sm * peak_per_sm,
                " %d SM%s x %d = %d FMA/cycle%s"
                % (n_sm, "" if n_sm == 1 else "s", peak_per_sm,
                   n_sm * peak_per_sm,
                   "" if used_sms >= n_sm or peak_sms
                   else "   (this run reached only %d of them)" % used_sms),
                transform=ax.get_yaxis_transform(), va="bottom",
                fontsize="small", color="gray")
    if log:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    ax.set_xlabel("threads (thread blocks x threads per block)")
    ax.set_ylabel("FMA / cycle")
    ax.grid(True, which="both", alpha=0.3)
    if any(series):
        ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()
