bases :=
bases += pd00_intro
bases += pd01_omp
bases += pd02_cuda
bases += pd03_omp_gpu
bases += pd04_simd_high_level
bases += pd05_simd_low_level
bases += pd06_ilp
bases += pd07_mem
bases += pd08_mm
bases += pd11_msort
bases += pd30_mlp

ipynb :

define nb_to_ipynb_rule
ipynb : notebooks/source/$(base)/$(base).ipynb
ipynb : notebooks/source/$(base)_ans/$(base)_ans.ipynb
notebooks/source/$(base)/dir notebooks/source/$(base)_ans/dir : % :
	mkdir -p $$@
notebooks/source/$(base)/$(base).ipynb : nb/source/$(base)/$(base).md notebooks/source/$(base)/dir
	uv run ./mk_nb.py --output $$@ --output-kernel py $$<
notebooks/source/$(base)_ans/$(base)_ans.ipynb : nb/source/$(base)/$(base).md notebooks/source/$(base)_ans/dir
	uv run ./mk_nb.py --output $$@ --output-kernel py $$< --label ans
endef


$(foreach base,$(bases),\
  $(eval $(call nb_to_ipynb_rule)))
