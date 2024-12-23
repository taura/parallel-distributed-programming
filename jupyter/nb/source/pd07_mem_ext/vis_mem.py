import re
import matplotlib.pyplot as plt

class parser:
    def __init__(self, a_txt):
        self.fp = open(a_txt)
        self.line = next(self.fp)
        self.header_p = re.compile(r"==== (?P<k>[^=]+)=(?P<v>\d+) ====")
        self.kv_p = re.compile(r"(?P<k>[^:]+) : [^\d]*(?P<v>\d+(\.\d+)?)")
        self.dics = []
    def next(self):
        try:
            self.line = next(self.fp)
        except StopIteration:
            self.line = ""
    def eat_header(self):
        matched = self.header_p.match(self.line)
        assert(matched), self.line
        k, v = matched.group("k", "v")
        self.next()
        self.dic[k] = float(v)
    def eat_kv(self):
        matched = self.kv_p.match(self.line)
        assert(matched), self.line
        k, v = matched.group("k", "v")
        self.next()
        self.dic[k] = float(v)
    def eat_checking_results_ok(self):
        assert(self.line == "checking results ... OK\n"), self.line
        self.next()
    def file(self):
        while self.header_p.match(self.line):
            self.result()
    def result(self):
        self.dic = {}
        self.eat_header()
        while self.kv_p.match(self.line):
            self.eat_kv()
        self.eat_checking_results_ok()
        self.dics.append(self.dic)

def parse(a_txt):
    psr = parser(a_txt)
    psr.file()
    return psr.dics

def vis_latency(files_txt):
    for a_txt in files_txt:
        dics = parse(a_txt)
        dics = sorted(dics, key=lambda d: d["sz"])
        X = [d["sz"] for d in dics]
        Y = [d["time_per_access"] for d in dics]
        plt.plot(X, Y, marker="*", label=a_txt)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("size (bytes)")
    plt.ylabel("latency (nsec)")
    plt.legend()
    plt.show()

def vis_bw(files_txt):
    for a_txt in files_txt:
        dics = parse(a_txt)
        dics = sorted(dics, key=lambda d: d["n_conc_cycles"])
        X = [d["n_conc_cycles"] for d in dics]
        Y = [d["bw"] for d in dics]
        plt.plot(X, Y, marker="*", label=a_txt)
        plt.plot(X, [x * Y[0] for x in X], label="ideal")
    plt.xlabel("concurrent chains")
    plt.ylabel("bandwidth (GB/sec)")
    plt.legend()
    plt.show()

def vis_bw_threads(files_txt):
    for a_txt in files_txt:
        dics = parse(a_txt)
        dics = sorted(dics, key=lambda d: d["n_cycles"])
        coaleses = sorted(list(set(d["coalese_size"] for d in dics)))
        for coalese in coaleses:
            dics_c = [d for d in dics if d["coalese_size"] == coalese]
            X = [d["n_cycles"] for d in dics_c]
            Y = [d["bw"] for d in dics_c]
            plt.plot(X, Y, marker="*", label=f"{a_txt} coalese={coalese}")
            if 0:
                plt.plot(X, [Y[0] * x / X[0] for x in X],
                         label=f"{a_txt} coalese={coalese} ideal")
    plt.xlabel("number of chains (= cuda threads)")
    plt.ylabel("bandwidth (GB/sec)")
    plt.ylim(0)
    plt.legend()
    plt.show()
    
# vis_latency(["include/a.txt", "include/a.txt"])
# vis_bw(["b.txt"])
# vis_bw_threads(["include/cuda.txt", "include/clang.txt"])
