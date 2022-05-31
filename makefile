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

coverage.xml:
	poetry run pytest --junitxml=junit.xml --cov=bicytok --cov-report xml:coverage.xml
