cpu_version_start:=1
cpu_version_end:=1
cuda_version_start:=1
cuda_version_end:=0
version_start:=$(cpu_version_start)
version_end:=$(cuda_version_start)

cc:=$(wildcard mem.cc)
h:=$(wildcard mem.h)

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
versioned/$(vcc)_$(v).exe : versioned/%.exe : versioned/%.cc versioned/mem_main_cpu.cc versioned/mem_cpu.h
#	clang++ -O3 -Wall -mavx512f -mfma -fopenmp -I. $$< versioned/mem_main_cpu.cc -o $$@ -lpfm
#	nvc++   -O3 -Wall -mavx512f -mfma -mp      -I. $$< versioned/mem_main_cpu.cc -o $$@ -lpfm
	g++     -O3 -Wall -mavx512f -mfma -fopenmp -I. $$< versioned/mem_main_cpu.cc -o $$@ -lpfm
versioned/$(vcc)_$(v).s : versioned/%.s : versioned/%.cc
#	clang++ -O3 -Wall -mavx512f -mfma -fopenmp -I. $$< -S -o $$@
#	nvc++   -O3 -Wall -mavx512f -mfma -mp      -I. $$< -S -o $$@
	g++     -O3 -Wall -mavx512f -mfma -fopenmp -I. $$< -S -o $$@
versioned/$(vcc)_$(v).cc : versioned/%_$(v).cc : %.cc $(versioned_dir)
	./mk_version.py -D VER=$(v) $$< > $$@
endef

define rule_cu
versioned/$(vcc)_$(v).exe : versioned/%.exe : versioned/%.cc versioned/mem_main_cuda.cc versioned/mem_cuda.h
	nvcc -O3 --generate-code arch=compute_80,code=sm_80 -I. -x cu $$< versioned/mem_main_cuda.cc -o $$@ -lpfm
versioned/$(vcc)_$(v).s : versioned/%.s : versioned/%.cc
	nvcc -O3 --generate-code arch=compute_80,code=sm_80 -I. -x cu $$< --ptx -o $$@
versioned/$(vcc)_$(v).cc : versioned/%_$(v).cc : $(vcc).cc $(versioned_dir)
	./mk_version.py -D VER=$(v) $$< > $$@
endef

versioned/mem_cpu.h : mem.h $(versioned_dir)
	./mk_version.py -D VER=1 $< > $@
versioned/mem_cuda.h : mem.h $(versioned_dir)
	./mk_version.py -D VER=2 $< > $@
versioned/mem_main_cpu.cc : mem_main.cc $(versioned_dir)
	./mk_version.py -D VER=1 $< > $@
versioned/mem_main_cuda.cc : mem_main.cc $(versioned_dir)
	./mk_version.py -D VER=2 $< > $@

$(foreach vcc,$(patsubst %.cc,%,$(cc)),\
$(foreach v,$(shell seq $(cpu_version_start) $(cpu_version_end)),\
$(eval $(call rule_cc))))

$(foreach vcc,$(patsubst %.cc,%,$(cc)),\
$(foreach v,$(shell seq $(cuda_version_start) $(cuda_version_end)),\
$(eval $(call rule_cu))))

$(versioned_dir) : % :
	mkdir -p $@
