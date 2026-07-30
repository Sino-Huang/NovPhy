## Rebuild

Compile each standalone source with either `latexmk`, `pdflatex`, or Tectonic.
For example:

```bash
tectonic 01_system_overview.tex
pdftocairo -svg 01_system_overview.pdf 01_system_overview.svg
pdftocairo -png -singlefile -r 180 01_system_overview.pdf 01_system_overview
```

The figures use the light Okabe-Ito OpenTikZ palette and parameterized spacing
near the top of each `.tex` file. Structural colors are limited to `otblue`,
`otorange`, `otteal`, `otpurple`, and `otgray`.
