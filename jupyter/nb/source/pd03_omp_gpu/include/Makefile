ext:=cc
#srcs:=$(wildcard *.$(ext))
srcs:=omp_gpu_sched_rec.cc
bases:=$(patsubst %.$(ext),%,$(srcs))
#mkv:=../../../../../../jupyter_tools/authoring/mk_version.py
mkv:=./mk_version.py

define rule
all : exes/$(base)_$(v)
vers/$(base)_$(v).$(ext) : $(base).$(ext) vers/dir
	$(mkv) -D VER=$(v) $$< > $$@
exes/$(base)_$(v) : vers/$(base)_$(v).$(ext) exes/dir
	nvc++ -mp=gpu -cuda -o $$@ $$<
endef

all :

$(foreach base,$(bases),$(foreach v,$(shell seq 1 $$($(mkv) --mode comment -D VER=0 $(base).$(ext))),$(eval $(call rule))))

exes/dir : 
	mkdir -p $@
vers/dir : 
	mkdir -p $@
clean :
	rm -rf exes vers

.DELETE_ON_ERROR:

# ccs:=$(wildcard *.cc)
# clang_exes:=$(patsubst %.cc,%.clang,$(ccs))
# nvc_exes:=$(patsubst %.cc,%.nvc,$(ccs))
# targets:=
# targets+=$(clang_exes)
# targets+=$(nvc_exes)

# all : $(targets)

# $(clang_exes) : %.clang : %.cc
# 	clang++ -o $@ -Wall -Wno-unknown-cuda-version -O3 -fopenmp -fopenmp-targets=nvptx64 $<

# $(nvc_exes) : %.nvc : %.cc
# 	nvc++ -o $@ -mp -cuda -O1 -target=gpu $<


