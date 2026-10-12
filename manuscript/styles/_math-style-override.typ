// Typst math mode defaults to italic variable letters and its own math font,
// and does not inherit `mainfont` from _quarto.yml (that's set inside
// Quarto's own template scope, which this header include sits outside of).
// Render inline math (e.g. `$\alpha$`) upright and in the body font to match
// surrounding text. Display/block equations (`$$...$$`) are left with
// Typst's default math styling for when real equations are added later.
// NOTE: keep the font name in sync with `mainfont` in _quarto.yml.
#show math.equation.where(block: false): it => {
  set text(font: "Times New Roman")
  math.upright(it)
}
