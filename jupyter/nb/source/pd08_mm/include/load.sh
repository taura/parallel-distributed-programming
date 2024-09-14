#!/bin/bash

db=a.sqlite
rm -f ${db}

txt2sql ${db} --table a \
        -e 'M = (?P<M>\d+), N = (?P<N>\d+), K = (?P<K>\d+)' \
        -e 'sizeof\(real\) = (?P<sz_real>\d+)' \
        -e 'repeat : (?P<repeat>\d+) times' \
        -e '^cycles : (?P<cycles>\d+)' \
        -r 'fmas/cycle' \
        out/out_*.txt

sqlite3 ${db} 'select M,N,K,repeat * M * N * K * 1.0 / cycles from a limit 10'
