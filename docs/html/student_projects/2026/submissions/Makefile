mds   := index.md $(wildcard */problem.md)
htmls := $(patsubst %.md,%.html,$(mds))

all : $(htmls)

$(htmls) : %.html : %.md
#	uv run pandoc --filter ../../../../scripts/filter.py --number-sections $< -o $@ 
	uv run pandoc --filter ../../../../scripts/filter.py $< -o $@ 

.DELETE_ON_ERROR:
