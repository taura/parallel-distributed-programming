#
# compile
#
#mk_nb_flags := --dbg 2

nb_srcs :=
nb_srcs += $(wildcard nb/source/pd??_*/*.sos nb/source/pd??_*/pd*.py)

aux_srcs :=
aux_srcs += $(wildcard nb/source/pd02_*/*.h)
aux_srcs += $(wildcard nb/source/pd03_*/omp_sched_rec.c)
aux_srcs += $(wildcard nb/source/pd03_*/aux.h)
aux_srcs += $(wildcard nb/source/pd03_*/log2ppm.py)
aux_srcs += $(wildcard nb/source/pd04_*/*.h)
#aux_srcs += $(wildcard nb/source/pd11_*/*.h)
#aux_srcs += $(wildcard nb/source/pd11_*/*.mk)
# aux_srcs += $(wildcard nb/source/pl04/*/*/*)
# aux_srcs += $(shell find nb/source/pl07/?? -type f)
# aux_srcs += $(shell find -L nb/source/pl10 -type f ! -name '*.sos')

users_csv := users.csv

feedback_class := pd
include ../../jupyter_tools/authoring/jupyter_tools.mk
