#!/bin/bash

db=a.sqlite
rm -f ${db}

./txt2sql ${db} --table a \
        -e 'size           : (?P<size>\d+) bytes' \
        -e 'stride         : (?P<stride>\d+)' \
        -e 'distinct lines : (?P<distinct_lines>\d+)' \
        -e 'accesses       : (?P<accesses>\d+)' \
        -e 'last element   : (?P<last>\d+)' \
        -e 'instructions : (?P<insns>\d+)' \
        -e 'cycles : (?P<cycles>\d+)' \
        -e 'L1-dcache-load-misses : (?P<l1_miss>\d+)' \
        -e 'l2_lines_in.all : (?P<l2_miss>\d+)' \
        -e 'offcore_requests.l3_miss_demand_data_rd : (?P<l3_miss>\d+)' \
        -r 'OK' \
        out/out_*.txt

sqlite3 ${db} 'select size,stride,cycles * 1.0 / accesses from a order by size limit 100'
