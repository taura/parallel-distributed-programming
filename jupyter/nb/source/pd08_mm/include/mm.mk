
MAX:=768
Ms:=8 16 32 64 $(shell seq 128 128 $(MAX))
Ns:=32 64 96 $(shell seq 160 128 $(MAX))
Ks:=144 208 $(shell seq 288 128 $(MAX))
Rs:=$(shell seq 0 4)

define rule
out/out_$(M)_$(N)_$(K)_$(R).txt : out/dir
	../versioned/pd08_mm/mm_4.exe $(M) $(N) $(K) > $$@
endef

targets:=$(foreach M,$(Ms),$(foreach N,$(Ns),$(foreach K,$(Ks),$(foreach R,$(Rs),out/out_$(M)_$(N)_$(K)_$(R).txt))))

all : $(targets)

$(foreach M,$(Ms),$(foreach N,$(Ns),$(foreach K,$(Ks),$(foreach R,$(Rs),$(eval $(call rule))))))

out/dir :
	mkdir -p $@
