ccs:=$(wildcard *.cc)
clang_exes:=$(patsubst %.cc,%.clang,$(ccs))
nvc_exes:=$(patsubst %.cc,%.nvc,$(ccs))
targets:=
targets+=$(clang_exes)
targets+=$(nvc_exes)

all : $(targets)

$(clang_exes) : %.clang : %.cc
	clang++ -o $@ -Wall -Wno-unknown-cuda-version -O3 -fopenmp -fopenmp-targets=nvptx64 $<

$(nvc_exes) : %.nvc : %.cc
	nvc++ -o $@ -mp -cuda -O1 -target=gpu $<

