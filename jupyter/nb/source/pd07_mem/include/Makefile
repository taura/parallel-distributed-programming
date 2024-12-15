# latency CPU GPU
#

dbg := 2
src_cc := latency.cc
bases := $(patsubst %.cc,%,$(src_cc))
vers :=
vers += omp			# single thread
#vers += simd
#vers += cuda

compilers := 
compilers += g++
compilers += clang++
compilers += nvc++
#compilers += nvcc

flags :=
flags += -DDBG=$(dbg)

clang++_flags :=
clang++_flags += -Wall
#clang++_flags += -O0 -g
clang++_flags += -O3
clang++_flags += -mfma
clang++_flags += -mavx512f

clang++_omp_flags += -fopenmp
clang++_flags += -fopenmp-targets=nvptx64 

nvc++_flags :=
#nvc++_flags += -O0 -g
nvc++_flags += -O3
nvc++_flags += -mp=gpu
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
exe/$(base)_$(compiler)_$(ver).exe : ver/$(base)_$(ver).cc exe/dir Makefile
	$(compiler) $(flags) $($(compiler)_flags) $($(compiler)_$(ver)_flags) -Dcompiler=\"$(compiler)\" -Dver=\"$(ver)\" -o $$@ $$<

asm/$(base)_$(compiler)_$(ver).s : ver/$(base)_$(ver).cc asm/dir Makefile
	$(compiler) $(flags) $($(compiler)_flags) $($(compiler)_$(ver)_flags) -Dcompiler="$(compiler)" -Dver="$(ver)" $($(compiler)_asm_flags) -o $$@ $$<
endef

# define compile_clang++
# exes += exe/$(base)_clang++.exe
# asms += asm/$(base)_clang++.s
# ver/$(base)_omp.cc : $(base).cc ver/dir Makefile
# 	./mk_version.py -DCFG="omp" -DDBG=0 $$< > $$@

# exe/$(base)_clang++.exe : ver/$(base)_omp.cc exe/dir Makefile
# 	clang++ $(flags) $(clang++_flags) -o $$@ $$<

# asm/$(base)_clang++.s : ver/$(base)_omp.cc asm/dir Makefile
# 	clang++ $(flags) $(clang++_flags) -S -o $$@ $$<
# endef

# define compile_simd_clang++
# exes += exe/$(base)_simd_clang++.exe
# asms += asm/$(base)_simd_clang++.s
# ver/$(base)_simd_omp.cc : $(base).cc ver/dir Makefile
# 	./mk_version.py -DCFG="simd_omp" -DDBG=0 $$< > $$@

# exe/$(base)_simd_clang++.exe : ver/$(base)_simd_omp.cc exe/dir Makefile
# 	clang++ $(flags) $(simd_clang++_flags) -o $$@ $$<

# asm/$(base)_simd_clang++.s : ver/$(base)_simd_omp.cc asm/dir Makefile
# 	clang++ $(flags) $(simd_clang++_flags) -S -o $$@ $$<
# endef

# define compile_nvc++
# exes += exe/$(base)_nvc++.exe
# asms += asm/$(base)_nvc++.s
# exe/$(base)_nvc++.exe : ver/$(base)_omp.cc exe/dir Makefile
# 	nvc++ $(flags) $(nvc++_flags) -o $$@ $$<

# asm/$(base)_nvc++.s : ver/$(base)_omp.cc asm/dir Makefile
# 	nvc++ $(flags) $(nvc++_flags) -S -o $$@ $$<
# endef

# define compile_simd_nvc++
# exes += exe/$(base)_simd_nvc++.exe
# asms += asm/$(base)_simd_nvc++.s
# exe/$(base)_simd_nvc++.exe : ver/$(base)_simd_omp.cc exe/dir Makefile
# 	nvc++ $(flags) $(simd_nvc++_flags) -o $$@ $$<

# asm/$(base)_simd_nvc++.s : ver/$(base)_simd_omp.cc asm/dir Makefile
# 	nvc++ $(flags) $(simd_nvc++_flags) -Mkeepasm -o $$@ $$<
# 	mv $(base)_simd_omp.s asm/
# endef

# define compile_nvcc
# exes += exe/$(base)_nvcc.exe
# asms += asm/$(base).ptx
# ver/$(base)_cuda.cc : $(base).cc ver/dir Makefile
# 	./mk_version.py -DCFG="cuda" -DDBG=0 $$< > $$@

# exe/$(base)_nvcc.exe : ver/$(base)_cuda.cc exe/dir  Makefile
# 	nvcc $(flags) $(nvcc_flags) -o $$@ -x cu $$<

# asm/$(base).ptx : ver/$(base)_cuda.cc asm/dir Makefile
# 	nvcc $(flags) $(nvcc_flags) --save-temps --keep-dir asm -o $$*.exe -x cu $$<
# endef

$(foreach base,$(bases),\
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
