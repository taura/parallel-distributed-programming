MAKEFLAGS += --warn-undefined-variables

md_files :=
#md_files += $(wildcard nb/source/foo/*.md)
#md_files += $(wildcard nb/source/pd00_intro/*.md)
md_files += $(wildcard nb/source/pd01_omp/pd??_**.md)
# md_files += pd02_cuda
# md_files += pd03_omp_gpu
# md_files += pd04_simd_high_level
# md_files += pd05_simd_low_level
# md_files += pd06_ilp
# md_files += pd07_mem
# md_files += pd08_mm
# md_files += pd11_msort
# md_files += pd30_mlp

aux_files := 
aux_files += $(wildcard nb/source/pd00_intro/svg/*.svg)
aux_files += $(wildcard nb/source/*/AGENTS.md)
aux_files += $(wildcard nb/source/*/ask.chat)
# aux_files += $(wildcard nb/source/pd01_omp/sched_vis.py)

EVAL=$(eval $(1))
#EVAL=$(info $(1))

generate :
ipynb :

define dir_rule
assignments/source/$(asgn)/dir assignments/source/$(asgn)_ans/dir : % :
	mkdir -p $$@
endef

define generate_rule
generate : assignments/release/$(asgn)/generated
generate : assignments/release/$(asgn)_ans/generated
assignments/release/$(asgn)/generated : % :
	uv run nbgrader generate_assignment $(asgn) --force --CourseDirectory.root=assignments
	touch $$@
assignments/release/$(asgn)_ans/generated : % :
	uv run nbgrader generate_assignment $(asgn)_ans --force --CourseDirectory.root=assignments
	touch $$@
endef

#
# markdown -> ipynb
#
define nb_to_ipynb_rule
ipynb : assignments/source/$(asgn)/$(file).ipynb
ipynb : assignments/source/$(asgn)_ans/$(file).ipynb
assignments/release/$(asgn)/generated : assignments/source/$(asgn)/$(file).ipynb
assignments/release/$(asgn)/generated : assignments/source/$(asgn)_ans/$(file).ipynb
assignments/source/$(asgn)/$(file).ipynb : nb/source/$(asgn)/$(file).md assignments/source/$(asgn)/dir
	mkdir -p $$(dir $$@)
	uv run ./mk_nb.py --output $$@ --output-kernel miyabi $$<
assignments/source/$(asgn)_ans/$(file).ipynb : nb/source/$(asgn)/$(file).md assignments/source/$(asgn)_ans/dir
	mkdir -p $$(dir $$@)
	uv run ./mk_nb.py --output $$@ --output-kernel miyabi $$< --label ans
endef

#
# split path into assignment and file name, e.g.
# pd00_intro/img/a.png -> pd00_intro  img/a.png
#
aux_file_asgn = $(firstword $(subst /, ,$(patsubst nb/source/%,%,$(aux_file))))
aux_file_file = $(patsubst $(call aux_file_asgn)/%,%,$(patsubst nb/source/%,%,$(aux_file)))
md_file_asgn = $(firstword $(subst /, ,$(patsubst nb/source/%,%,$(asgn_md))))
md_file_file = $(patsubst $(call md_file_asgn)/%.md,%,$(patsubst nb/source/%,%,$(asgn_md)))

#
# copy aux files
#
define aux_file_rule
ipynb : assignments/source/$(asgn)/$(file)
ipynb : assignments/source/$(asgn)_ans/$(file)
assignments/release/$(asgn)/generated : assignments/source/$(asgn)/$(file)
assignments/release/$(asgn)/generated : assignments/source/$(asgn)_ans/$(file)
assignments/source/$(asgn)/$(file) : nb/source/$(asgn)/$(file) assignments/source/$(asgn)/dir
	install -D --mode 0644 $$< $$@
assignments/source/$(asgn)_ans/$(file) : nb/source/$(asgn)/$(file) assignments/source/$(asgn)_ans/dir
	install -D --mode 0644 $$< $$@
endef

assignments := $(sort $(foreach asgn_md,$(md_files),$(md_file_asgn)))
$(foreach asgn,$(assignments),\
  $(call EVAL,$(call dir_rule)))

$(foreach asgn_md,$(md_files),\
  $(let asgn file,$(md_file_asgn) $(md_file_file),\
    $(call EVAL,$(call nb_to_ipynb_rule))))

$(foreach aux_file,$(aux_files),\
  $(let asgn file,$(aux_file_asgn) $(aux_file_file),\
    $(call EVAL,$(call aux_file_rule))))

$(foreach asgn,$(assignments),\
  $(call EVAL,$(call generate_rule)))


.PHONY : ipynb generate 
