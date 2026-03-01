# TrialAgent IEEE Report (LaTeX)

## Files
- `main.tex` — IEEE-format report source
- `references.bib` — BibTeX references

## Build
Use a TeX distribution with `IEEEtran` available.

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes
- The report content is grounded in the current TrialAgent implementation.
- Citations include requested related systems and the data/tool APIs used in this project.
