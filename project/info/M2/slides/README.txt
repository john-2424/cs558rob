CS558IRL Project Milestone 2 -- Presentation Slides
=====================================================

Contents
--------
  main.tex        Beamer (metropolis theme) source for the 18-slide deck
                  plus 5 backup slides in the appendix.
  assets/         PNGs referenced by main.tex.  The eight plots are
                  direct copies from project/results/m2/plots/.  The
                  four demo thumbnail placeholders are drawn inline
                  as text boxes; replace them with real frame grabs
                  when packaging the submission.

Compile options
---------------

1) Overleaf (recommended)
     - Zip this entire "slides/" folder.
     - Upload the zip to Overleaf ("New Project" -> "Upload Project").
     - Set main document to main.tex; compile.
     - The metropolis theme is preinstalled on Overleaf; no extra
       packages needed.

2) Local compile
     $ latexmk -pdf main.tex
   Requires: pdflatex, metropolis theme.  If metropolis is missing,
   edit main.tex and change
       \usetheme{metropolis}
   to
       \usetheme{Madrid}
   No other edits are required.

Replacing the demo thumbnails
-----------------------------
Slides 14 and 15 currently use text placeholders for the four canonical
demo videos located (relative to the repo root) at

    ../../Project/Milestone 2/

The four canonical recordings (confirmed):
    Planner Only Perturb xy0.0 z0.0 yaw0.0.mp4
    Planner Only Perturb xy0.04 z0.005 yaw0.10.mp4
    Hybrid Perturb xy0.04 z0.005 yaw0.10.mp4
    RL Only Perturb xy0.04 z0.005 yaw0.10.mp4

Extract one frame per video with ffmpeg (install first if missing, e.g.
`choco install ffmpeg` on Windows / `brew install ffmpeg` on mac):

    cd project/info/M2/slides/assets
    ffmpeg -i "../../../../../Project/Milestone 2/Planner Only Perturb xy0.04 z0.005 yaw0.10.mp4" \
           -ss 00:00:04 -vframes 1 demo_thumb_planner_004.png
    ffmpeg -i "../../../../../Project/Milestone 2/Hybrid Perturb xy0.04 z0.005 yaw0.10.mp4" \
           -ss 00:00:06 -vframes 1 demo_thumb_hybrid_004.png
    ffmpeg -i "../../../../../Project/Milestone 2/Planner Only Perturb xy0.0 z0.0 yaw0.0.mp4" \
           -ss 00:00:04 -vframes 1 demo_thumb_planner_nominal.png
    ffmpeg -i "../../../../../Project/Milestone 2/RL Only Perturb xy0.04 z0.005 yaw0.10.mp4" \
           -ss 00:00:06 -vframes 1 demo_thumb_rlonly_004.png

Then inside main.tex on slides 14 and 15 uncomment the four
\includegraphics lines and remove the adjacent \fbox placeholders.

Numeric audit
-------------
Every number in the slide body is drawn from
  project/results/m2/eval_results.json
Any edit to the eval must be mirrored in main.tex.  Key callouts to
keep in sync:
  - planner success row: 1.00 / 0.46 / 0.45 / 0.29 / 0.19 / 0.25 / 0.22
  - hybrid success row:  0.78 / 0.65 / 0.79 / 0.85 / 0.78 / 0.77 / 0.71
  - rl_only success:     0.00 everywhere
  - hybrid mean residual:  0.050--0.055 rad
  - hybrid mean EE-cube:   0.018--0.026 m
  - planner mean EE-cube:  0.020 m -> 0.056 m
  - rl_only mean EE-cube:  0.096--0.121 m

Slide count and pacing
----------------------
  18 content slides + 5 appendix = 23 total.
  10-minute target => ~33 seconds per content slide.
  Slides 6 (architecture), 9 (iteration), 11 (headline), 13 (ablation)
  usually run long.  Rehearse those first.

Team placeholders
-----------------
Replace "[Team Member A]" / "[Team Member B]" in main.tex (author
field on the title slide, and Contributions column on slide 17) with
actual names/roles before the final recording.
