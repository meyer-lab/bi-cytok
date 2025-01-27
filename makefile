SHELL := /bin/bash

flist = $(wildcard bicytok/figures/figure*.py)

.PHONY: clean test all testprofile pyright

all: $(patsubst bicytok/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: bicytok/figures/figure%.py
	@ mkdir -p ./output
	rye run fbuild $*

clean:
	rm -r output

test: .venv
	rye run pytest -s -v -x

.venv: pyproject.toml
	rye sync

testprofile:
	rye run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

coverage.xml: .venv
	rye run pytest --junitxml=junit.xml --cov=bicytok --cov-report xml:coverage.xml

pyright: .venv
	rye run pyright bicytok
