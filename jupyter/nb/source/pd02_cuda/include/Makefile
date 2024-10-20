ext:=cu
#srcs:=$(wildcard *.$(ext))
srcs:=cuda_sched_rec.cu
bases:=$(patsubst %.cu,%,$(srcs))
mkv:=../../../../../../jupyter_tools/authoring/mk_version.py

define rule
all : exes/$(base)_$(v)
vers/$(base)_$(v).$(ext) : $(base).$(ext) vers/dir
	$(mkv) -D VER=$(v) $$< > $$@
exes/$(base)_$(v) : vers/$(base)_$(v).$(ext) exes/dir
	nvcc -I . -o $$@ $$<
endef

all :

$(foreach base,$(bases),$(foreach v,$(shell seq 1 $$($(mkv) --mode comment -D VER=0 $(base).$(ext))),$(eval $(call rule))))

exes/dir : 
	mkdir -p $@
vers/dir : 
	mkdir -p $@
clean :
	rm -rf exes vers
