#!/bin/bash
# run.sh : build (if needed) and run the IVF baseline on Miyabi
#
# usage (from a notebook cell, after `cd code`):
#   bash run.sh                 # default problem size
#   bash run.sh small           # quick check (a few seconds)
#   bash run.sh -n 500000 -p 8  # any options of ./ivf are passed through
#
# Run it with %%sub so that it executes on a compute node, not the login node.
set -euo pipefail
cd "$(dirname "$0")"

export PATH=/work/gt81/share/env/llvm/bin:$PATH
# the nvidia module exports CXX=nvc++, so use our own variable names
CXX=${IVF_CXX:-clang++}
CXXFLAGS=${IVF_CXXFLAGS:-"-O3 -Wall -std=c++17"}

if [ ! -x ivf ] || [ ivf -ot main.cc ] || [ ivf -ot ivf.cc ] || [ ivf -ot ivf.h ]; then
  echo "building: $CXX $CXXFLAGS main.cc ivf.cc -o ivf"
  $CXX $CXXFLAGS main.cc ivf.cc -o ivf
fi

echo "host: $(hostname)  date: $(date '+%F %T')"
echo "cpu : $(lscpu | grep -m1 'Model name' | sed 's/.*: *//')"
echo "cxx : $(command -v $CXX || echo "$CXX not found") $($CXX --version 2>&1 | head -1 || true)"

if [ "${1:-}" = "small" ]; then
  shift
  set -- -n 100000 -q 1000 -l 256 -t 20000 -p 8 -g 50 "$@"
fi

rc=0
./ivf "$@" || rc=$?
if [ $rc -ne 0 ]; then
  echo "ivf exited with status $rc (137 = killed, e.g. out of memory or a limit on the login node)"
fi
exit $rc
