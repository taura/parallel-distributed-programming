
merge_version_start:=1
merge_version_end=$(shell ./mk_version.py --mode comment -DVER=0 merge.cc)
msort_version_start:=1
msort_version_end=$(shell ./mk_version.py --mode comment -DVER=0 msort.cc)

merge_versioned_cc:=\
$(foreach v,$(shell seq $(merge_version_start) $(merge_version_end)),\
versioned/merge_$(v).cc)

msort_versioned_cc:=\
$(foreach u,$(shell seq $(msort_version_start) $(msort_version_end)),\
$(foreach v,$(shell seq $(merge_version_start) $(merge_version_end)),\
versioned/msort_$(u)_$(v).cc))

versioned_dir:=versioned/dir

exe:=$(patsubst versioned/%.cc,versioned/%.exe,$(merge_versioned_cc) $(msort_versioned_cc))
asm:=$(patsubst versioned/%.cc,versioned/%.s,$(merge_versioned_cc) $(msort_versioned_cc))

targets:=
targets+=$(exe)
#targets+=$(asm)

all : $(targets)

define rule_merge_cc
versioned/merge_$(v).cc : versioned/%_$(v).cc : %.cc $(versioned_dir)
	./mk_version.py -D VER=$(v) $$< > $$@
endef

$(foreach v,$(shell seq $(merge_version_start) $(merge_version_end)),\
$(eval $(call rule_merge_cc)))

define rule_msort_cc
versioned/msort_$(v).cc : versioned/%_$(v).cc : %.cc $(versioned_dir)
	./mk_version.py -D VER=$(v) $$< > $$@
endef

$(foreach v,$(shell seq $(msort_version_start) $(msort_version_end)),\
$(eval $(call rule_msort_cc)))

define rule_merge_exe
versioned/merge_$(v).exe : versioned/%.exe : versioned/merge_$(v).cc merge_main.cc msort.h
	clang++ -O3 -Wall -fopenmp -I. $$< merge_main.cc -o $$@ -lpfm
#	nvc++ -O3 -Wall -mp -I. $$< merge_main.cc -o $$@ -lpfm
versioned/merge_$(v).s : versioned/%.s : versioned/%.cc msort.h
	clang++ -O3 -Wall -fopenmp -I. $$< -S
#	nvc++ -O3 -Wall -fopenmp -I. $$< -S
endef

$(foreach v,$(shell seq $(merge_version_start) $(merge_version_end)),\
$(eval $(call rule_merge_exe)))

define rule_msort_exe
versioned/msort_$(u)_$(v).exe : versioned/%.exe : versioned/msort_$(u).cc versioned/merge_$(v).cc msort_main.cc msort.h
	clang++ -O3 -Wall -fopenmp -I. $$< versioned/merge_$(v).cc msort_main.cc -o $$@ -lpfm
#	nvc++ -O3 -Wall -mp -I. $$+ -o $$@ -lpfm
versioned/msort_$(u)_$(v).s : versioned/%.s : versioned/%.cc msort.h
	clang++ -O3 -Wall -fopenmp -I. $$< -S
#	nvc++ -O3 -Wall -mp -I. $$< -S
endef

$(foreach u,$(shell seq $(msort_version_start) $(msort_version_end)),\
$(foreach v,$(shell seq $(merge_version_start) $(merge_version_end)),\
$(eval $(call rule_msort_exe))))

$(versioned_dir) : % :
	mkdir -p $@
