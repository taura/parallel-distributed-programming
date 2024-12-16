import re
import matplotlib.pyplot as plt

class parser:
    def __init__(self, a_txt):
        self.fp = open(a_txt)
        self.line = next(self.fp)
        self.header_p = re.compile(r"==== m=(?P<m>\d+) ====")
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
        m = int(matched.group("m"))
        self.next()
        self.dic["m"] = m
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
        X = [d["sz"] for d in dics]
        Y = [d["latency_per_elem"] for d in dics]
        plt.plot(X, Y, marker="-o-", label=a_txt)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("size (bytes)")
    plt.ylabel("latency (nsec)")
    plt.legend()
    plt.show()
    
# vis_latency(["include/a.txt", "include/a.txt"])
