SHELL := /bin/bash

flist = 1 2 3

.PHONY: clean test all testprofile testcover spell

all: pylint.log $(patsubst %, output/figure%.svg, $(flist))

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: venv genFigures.py ckine/figures/figure%.py
	mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

clean:
	mv output/requests-cache.sqlite requests-cache.sqlite || true
	rm -rf prof output coverage.xml .coverage .coverage* junit.xml coverage.xml profile profile.svg pylint.log
	mkdir output
	mv requests-cache.sqlite output/requests-cache.sqlite || true
	rm -f profile.p* stats.dat .coverage nosetests.xml coverage.xml testResults.xml
	rm -rf html doxy.log graph_all.svg venv ./ckine/data/flow
	find -iname "*.pyc" -delete

download: .dataURLs.txt
	mkdir -p ./ckine/data/flow
	wget -nv -N -P ./ckine/data/flow/ -i .dataURLs.txt
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2019-03-15 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - NK plate.zip'
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2019-04-18 IL-2 and IL-15 treated pSTAT5 assay - Lymphocyte gated - Treg plate - NEW PBMC LOT.zip'
	unzip -qd ./ckine/data/flow/ './ckine/data/flow/2021.04.27 Peter org PBMC signaling and receptor.zip'
	tar -x --strip-components=1 -C ./ckine/data/flow/ -f ./ckine/data/flow/receptor-profiling-20200909.tar.xz

test: venv
	. venv/bin/activate && pytest

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov-branch --cov=ckine --cov-report xml:coverage.xml

pylint.log: venv common/pylintrc
	. venv/bin/activate && (pylint --rcfile=./common/pylintrc ckine > pylint.log || echo "pylint3 exited with $?")
