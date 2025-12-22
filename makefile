SHELL := /bin/bash

flist = $(wildcard bicytok/figures/figure*.py)

.PHONY: clean test all testprofile pyright ruff-check-figures ruff-format-figures ruff-check-all ruff-format-all

all: $(patsubst bicytok/figures/figure%.py, output/figure%.svg, $(flist))

output/figure%.svg: bicytok/figures/figure%.py
	@ mkdir -p ./output
	uv run fbuild $*

clean:
	rm -r output

test: .venv
	uv run pytest -s -v -x

.venv: pyproject.toml
	uv sync --dev

testprofile:
	uv run python3 -m cProfile -o profile -m pytest -s -v -x
	gprof2dot -f pstats --node-thres=5.0 profile | dot -Tsvg -o profile.svg

coverage.xml: .venv
	uv run pytest --junitxml=junit.xml --cov=bicytok --cov-report xml:coverage.xml

pyright: .venv
	uv run pyright bicytok

# Ruff linting and formatting for Quarto figures
ruff-check-figures: .venv
	uv run python bicytok/lint_qmd.py check figures

ruff-format-figures: .venv
	uv run python bicytok/lint_qmd.py format figures

# Ruff linting and formatting for all code (including Quarto)
ruff-check-all: .venv
	uv run ruff check bicytok
	uv run python bicytok/lint_qmd.py check figures

ruff-format-all: .venv
	uv run ruff format bicytok
	uv run python bicytok/lint_qmd.py format figures
