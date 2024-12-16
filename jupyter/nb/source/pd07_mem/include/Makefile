# latency CPU GPU
#

dbg := 0
src_cc := latency.cc
bases := $(patsubst %.cc,%,$(src_cc))
vers :=
vers += omp
vers += omp_ilp
vers += omp_ilp_c
#vers += simd
#vers += cuda

compilers := 
#compilers += g++
compilers += clang++
#compilers += nvc++
#compilers += nvcc

flags :=
flags += -DDBG=$(dbg)

clang++_flags :=
clang++_flags += -Wall
#clang++_flags += -O0 -g
clang++_flags += -O3
#clang++_flags += -mfma
#clang++_flags += -mavx512f
clang++_flags += -march=native
clang++_flags += -Wno-vla-cxx-extension

clang++_omp_flags += -fopenmp
#clang++_flags += -fopenmp-targets=nvptx64 

nvc++_flags :=
#nvc++_flags += -O0 -g
nvc++_flags += -O3
#nvc++_flags += -mp=gpu
nvc++_flags += -mp=multicore
#nvc++_flags += -gpu=mem:managed
nvc++_flags += -cuda
nvc++_flags += -march=native

nvcc_flags :=
#nvcc_flags += -O0 -g
nvcc_flags += -O4

exes :=
asms :=

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

define gen_ver
ver/$(base)_$(ver).cc : $(base).cc ver/dir Makefile
	./mk_version.py -DVER="$(ver)" -DDBG=$(dbg) $$< > $$@.t
	chmod 444 $$@.t
	mv -f $$@.t $$@
endef

define compile
exes += exe/$(base)_$(compiler)_$(ver).exe
#asms += asm/$(base)_$(compiler)_$(ver).s
exe/$(base)_$(compiler)_$(ver).exe : ver/$(base)_$(ver).cc ver/main_$(ver).cc exe/dir Makefile
	$(compiler) $(flags) $($(compiler)_flags) $($(compiler)_$(ver)_flags) -o $$@ ver/$(base)_$(ver).cc ver/main_$(ver).cc
# -Dcompiler=\"$(compiler)\" -Dver=\"$(ver)\" 

asm/$(base)_$(compiler)_$(ver).s : ver/$(base)_$(ver).cc asm/dir Makefile
	$(compiler) $(flags) $($(compiler)_flags) $($(compiler)_$(ver)_flags) $($(compiler)_asm_flags) -S -o $$@ $$<
# -Dcompiler="$(compiler)" -Dver="$(ver)" 
endef

$(foreach base,$(bases) main,\
$(foreach ver,$(vers),\
$(eval $(call gen_ver))))

$(foreach base,$(bases),\
$(foreach ver,$(vers),\
$(foreach compiler,$(compilers),\
$(eval $(call compile)))))

exe/dir asm/dir ver/dir : % :
	mkdir -p $@

all : $(exes) $(asms)

clean :
	rm -rf *.ii *.cudafe1.c *.cudafe1.cpp *.cudafe1.gpu *.cudafe1.stub.c *.fatbin *.fatbin.c *.o *.ptx *.cubin *.reg.c *.module_id asm exe ver
