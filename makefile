SHELL := /bin/bash

flist = $(wildcard bicytok/figures/figure*.py)

.PHONY: clean test all testprofile testcover spell

all: $(patsubst bicytok/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: bicytok/figures/figure%.py
	mkdir -p ./output
	poetry run fbuild $*

clean:
	rm -r output

test:
	poetry run pytest -s -v -x

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

coverage.xml:
	poetry run pytest --junitxml=junit.xml --cov=bicytok --cov-report xml:coverage.xml
