# index.md = teams.md exported from adm/2026/teaming/teaming.html ("Export for students")
all : index.html

index.html : index.md
	uv run pandoc --filter ../../../../scripts/filter.py $< -o $@

.DELETE_ON_ERROR:
