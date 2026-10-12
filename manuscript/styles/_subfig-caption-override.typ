// Typst override for Quarto subfigure rendering (mirrors quarto_super from the
// Quarto 1.10.x typst template, with one change).
//
// Purpose: suppress the floating "(a)/(b)/..." enumerator that Quarto prints
// under each subpanel. Subfigure *numbering* is untouched, so cross-references
// (e.g. @fig-scale-emd -> "Figure 2h") still resolve. This lets every figure use
// a single combined caption on the parent (with (a), (b), ... described inline)
// with no redundant floating letters under the panels.
//
// NOTE: this redefines quarto_super and must be kept in sync if Quarto's typst
// template changes. The only functional edit vs. the stock definition is the
// inner `show figure.caption` rule below (which drops the panel caption entirely).
#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => {
          let subfloat-idx = quartosubfloatcounter.get().first() + 1
          subfloat-numbering(n-super, subfloat-idx)
        })
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          show figure.caption: _ => []
          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}
