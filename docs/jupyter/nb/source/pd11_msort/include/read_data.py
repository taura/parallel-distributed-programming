#!/usr/bin/python3
import re
import pandas as pd

def mk_key(s):
    s = s.strip()
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    return s

def mk_val(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip()

def read_file(filename):
    data = {}
    pat = re.compile("(?P<key>[^:]+):(?P<val>[^:]+)")
    with open(filename) as fp:
        for line in fp:
            m = pat.match(line)
            if m:
                key = mk_key(m.group("key"))
                val = mk_val(m.group("val"))
                data[key] = val
    return data

def read_files(filenames):
    data = []
    for filename in filenames:
        data.append(read_file(filename))
    return pd.DataFrame(data)


