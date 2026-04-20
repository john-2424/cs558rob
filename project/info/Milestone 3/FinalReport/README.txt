CS558IRL Project - Milestone 3 Final Report
============================================

Contents
--------
  FinalReport.tex   IEEEtran journal source, single-file, hard-coded bibliography
  Reference.bib     BibTeX entries (kept alongside for reference; the .tex uses
                    an inline thebibliography block so no bibtex pass is required)
  IEEEtran.cls      IEEE LaTeX class file (copied from the midterm folder)
  figures/          PNGs referenced by FinalReport.tex; copies of the eval
                    outputs in project/results/m2/plots/

Compile options
---------------

1) Overleaf (recommended)
     - Zip the FinalReport/ folder.
     - Upload on Overleaf ("New Project" -> "Upload Project").
     - Set main document = FinalReport.tex; compile with pdfLaTeX.
     - No bibtex pass is needed (bibliography is inline).

2) Local compile
     $ latexmk -pdf FinalReport.tex
   or:
     $ pdflatex FinalReport.tex (twice, for cross-refs)

Rubric mapping (project/info/Milestone 3/FinalReport.pdf)
---------------------------------------------------------
  Abstract (10 pts)                    -- single paragraph, above Section I
  I. Introduction & Motivation (10 pts) -- 3 paragraphs, citations present
  II. Problem Formulation (15 pts)      -- definitions + 5 numbered equations
                                          + \begin{problem} block
  III. Main Results (40 pts)           -- 4 subsections: backbone, architecture,
                                          iteration, evaluation protocol
  IV. Simulations (20 pts)             -- 4 figures (success, residual, phase,
                                          grasp) + summary table
  V. Conclusion (5 pts)                -- 1 paragraph
  References                           -- 12 IEEE-style entries inline

Numeric audit
-------------
Every decimal in FinalReport.tex is drawn from
  project/results/m2/eval_results.json
If the eval is ever rerun, mirror these cells:
  - planner success row: 1.00, 0.46, 0.45, 0.29, 0.19, 0.25, 0.22
  - hybrid success row:  0.78, 0.65, 0.79, 0.85, 0.78, 0.77, 0.71
  - rl_only success:     0.00 at every level
  - hybrid mean residual range: 0.0496 - 0.0549 rad
  - hybrid mean EE-cube distance range: 0.018 - 0.026 m
  - planner mean EE-cube distance: 0.020 m -> 0.056 m
  - rl_only mean EE-cube distance: 0.096 - 0.121 m

Reference.bib
-------------
Kept for future edits in case you switch to a \bibliography{Reference} pass.
The inline bibitems in FinalReport.tex and the entries in Reference.bib
use the same citation keys, so a switch is a one-line change at the
bottom of the .tex.
