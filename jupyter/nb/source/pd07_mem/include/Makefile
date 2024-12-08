src_cc := latency.cc
bases := $(patsubst %.cc,%,$(src_cc))

compilers := clang++

clang++_flags :=
clang++_flags += -Wall
clang++_flags += -O3

.DEFAULT_GOAL := exe

define compile_clang++
exes += exe/$(base)_clang++.exe
exe/$(base)_clang++.exe : $(base).cc exe/dir
	clang++ $(clang++_flags) -fopenmp -fopenmp-targets=nvptx64 -o $$@ $$<
endef

$(foreach base,$(bases),\
$(foreach compiler,$(compilers),\
$(eval $(call compile_$(compiler)))))

exe/dir :
	mkdir -p $@

exe : $(exes)
