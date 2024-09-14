cpu_version_start:=1
cpu_version_end:=9
cuda_version_start:=10
cuda_version_end:=10
version_start:=$(cpu_version_start)
version_end:=$(cuda_version_start)

cc:=$(wildcard mm.cc)
h:=$(wildcard mm.h)

versioned_cc:=\
$(foreach vcc,$(patsubst %.cc,%,$(cc)),\
$(foreach v,$(shell seq $(version_start) $(version_end)),\
versioned/$(vcc)_$(v).cc))

versioned_h:=\
$(foreach vh,$(patsubst %.h,%,$(h)),\
$(foreach v,$(shell seq 1 $(shell ./mk_version.py --mode comment -DVER=0 $(vh).h)),\
versioned/$(vh)_$(v).h))

versioned_dir:=versioned/dir

exe:=$(patsubst versioned/%.cc,versioned/%.exe,$(versioned_cc))
asm:=$(patsubst versioned/%.cc,versioned/%.s,$(versioned_cc))

targets:=
targets+=$(exe)
targets+=$(asm)

all : $(targets)

define rule_cc
versioned/$(vcc)_$(v).exe : versioned/%.exe : versioned/%.cc versioned/mm_main_cpu.cc versioned/mm_cpu.h
#	clang++ -O3 -Wall -mavx512f -mfma -fopenmp -I. $$< versioned/mm_main_cpu.cc -o $$@ -lpfm
#	nvc++   -O3 -Wall -mavx512f -mfma -mp      -I. $$< versioned/mm_main_cpu.cc -o $$@ -lpfm
	g++     -O3 -Wall -mavx512f -mfma -fopenmp -I. $$< versioned/mm_main_cpu.cc -o $$@ -lpfm
versioned/$(vcc)_$(v).s : versioned/%.s : versioned/%.cc
#	clang++ -O3 -Wall -mavx512f -mfma -fopenmp -I. $$< -S -o $$@
#	nvc++   -O3 -Wall -mavx512f -mfma -mp      -I. $$< -S -o $$@
	g++     -O3 -Wall -mavx512f -mfma -fopenmp -I. $$< -S -o $$@
versioned/$(vcc)_$(v).cc : versioned/%_$(v).cc : %.cc $(versioned_dir)
	./mk_version.py -D VER=$(v) $$< > $$@
endef

define rule_cu
versioned/$(vcc)_$(v).exe : versioned/%.exe : versioned/%.cc versioned/mm_main_cuda.cc versioned/mm_cuda.h
	nvcc -O3 --generate-code arch=compute_80,code=sm_80 -I. -x cu $$< versioned/mm_main_cuda.cc -o $$@ -lpfm
versioned/$(vcc)_$(v).s : versioned/%.s : versioned/%.cc
	nvcc -O3 --generate-code arch=compute_80,code=sm_80 -I. -x cu $$< --ptx -o $$@
versioned/$(vcc)_$(v).cc : versioned/%_$(v).cc : $(vcc).cc $(versioned_dir)
	./mk_version.py -D VER=$(v) $$< > $$@
endef

#	nvc++ -gpu=cc80 -Wall $$< -o $$@
#	clang++ -Wall -Wno-unused-function -Wno-unknown-cuda-version --cuda-gpu-arch=sm_80 $$< -o $$@ -L/usr/local/cuda/lib64 -lcudart

versioned/mm_cpu.h : mm.h $(versioned_dir)
	./mk_version.py -D VER=1 $< > $@
versioned/mm_cuda.h : mm.h $(versioned_dir)
	./mk_version.py -D VER=2 $< > $@
versioned/mm_main_cpu.cc : mm_main.cc $(versioned_dir)
	./mk_version.py -D VER=1 $< > $@
versioned/mm_main_cuda.cc : mm_main.cc $(versioned_dir)
	./mk_version.py -D VER=2 $< > $@

$(foreach vcc,$(patsubst %.cc,%,$(cc)),\
$(foreach v,$(shell seq $(cpu_version_start) $(cpu_version_end)),\
$(eval $(call rule_cc))))

$(foreach vcc,$(patsubst %.cc,%,$(cc)),\
$(foreach v,$(shell seq $(cuda_version_start) $(cuda_version_end)),\
$(eval $(call rule_cu))))

$(versioned_dir) : % :
	mkdir -p $@
