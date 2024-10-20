ext:=cc
#srcs:=$(wildcard *.$(ext))
srcs:=omp_gpu_integral.cc
bases:=$(patsubst %.$(ext),%,$(srcs))
#mkv:=../../../../../../jupyter_tools/authoring/mk_version.py
mkv:=./mk_version.py

define rule
all : exes/$(base)_$(v)
vers/$(base)_$(v).$(ext) : $(base).$(ext) vers/dir
	$(mkv) -D VER=$(v) $$< > $$@
exes/$(base)_$(v) : vers/$(base)_$(v).$(ext) exes/dir
	nvc++ -mp=gpu -cuda -O4 -o $$@ $$<
#	clang++ -fopenmp -fopenmp-targets=nvptx64 -O4 -o $$@ $$<
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

