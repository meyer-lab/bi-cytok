1. File naming should be consistent. The convention for python files is lower
case letters with `_` to separate words. E.g., `BindingMod.py` ->
`binding_mod.py`. 
2. Sort imports with Ruff (I did this for you)
3. Run `ruff check` again with the rules I declared in `pyproject.toml`. There
are a few small issues outlined in the output.
4. When possible, use `snake_case` for variable and function naming. Officially,
the PEP8 convention specifies that variable names should only include lower case
letters, but it makes sense for us to ignore that part of the convention
sometimes, particularly when we're implementing mathematical functions which
were described in a math paper elsewhere, where uppercase variable names were
used (e.g. binding model paper uses R_eq).
5. I believe `bicytok/data/epitopeSelectivityList.csv`,
`bicytok/data/MonomericMutSingleCellData.csv`,
`bicytok/data/WTDimericMutSingleCellData.csv`, `bicytok/data/WTmutAffData.csv`
are unused.
