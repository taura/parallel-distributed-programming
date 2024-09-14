#  16 24 32 40 64 72 128 136 256 264 512 520 1024 1032 2048 2056 4096 4104
strides := $(shell (./seq.py 64 8192 2 64 0 ; ./seq.py 64 8192 1.4 64 1) | sort -n | uniq)
sizes := $(shell ./seq.py 400 100000000 1.5 64 1)
tries := $(shell seq -w 1 1)

targets :=
targets += $(foreach size,$(sizes),\
	$(foreach stride,$(strides),\
	$(foreach try,$(tries),\
	out/out_$(size)_$(stride)_$(try).txt)))

all : $(targets)

define rule
out/out_$(size)_$(stride)_$(try).txt : out/dir
	./versioned/mem_1.exe -n $(size) -s $(stride) -a 10000000 > $$@
endef

out/dir :
	mkdir -p $@

$(foreach size,$(sizes),\
$(foreach stride,$(strides),\
$(foreach try,$(tries),\
$(eval $(call rule)))))
