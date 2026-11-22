# Pass 3 working state (read this first after any context reset)

Plan: `PLAN_V3.md`. Why: `REVIEW.md`. User said "start", then "continue, don't change anything".
Do NOT change the approach. Keep going down the list below without asking.

## Method (proven on the head pilot, see `out/hp3_head.png`)

1. Label source faces: seeds in `survey/seeds/<region>.json` -> `python -m retopo.seeds grow`
   -> `survey/labels.npz` (3 s for the whole model). View labels:
   `python -m retopo.seeds view out/x.png cx cy cz half front,left,back [--ticks] [--labels f]`.
   Seed = [view, a, b(, r)]: first surface hit seen from that view (front/back: x,y; left/right: z,y;
   top/bottom: x,z; "bottom>0.9" limits y). Borders settle on creases (`retopo/segment.py`).
2. Build each object from its segment: `python -m retopo.fromseg <name> --edge 3 [--levels 1] [--sharp]
   [--cover a,b] [--labels f]` -> `hipoly3/<name>.obj`, cage in `hp3_cages/`, numbers in `out/hp3/<name>.json`.
   (decimate -> QuadriFlow quads -> subdivide -> project onto own segment -> border skirt.)
   PASS = seg->obj p95 <= 1 mm, obj->seg p95 <= 1 mm, 100% quads, 0 non-manifold, 0 winding errors.
3. Look: `python -m scripts.review3 out/x.png cx cy cz half front,fl,left,back hipoly3 [names]`
   (ours | source). Whole-model numbers: `scripts/review_deviation.py` (edit dir to hipoly3).

Python: /tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python
(if gone: python3.13 venv, pip install bpy numpy scipy pillow trimesh fast-simplification). Run as modules from repo root.

## Done

- Review + revised plan committed. Tooling: `retopo/segment.py`, `seeds.py`, `qremesh.py`, `fromseg.py`,
  `scripts/review_compare.py`, `review_deviation.py`, `review3.py`.
- Head region seeds (`survey/seeds/head.json`): hat_crown, hat_band, hat_brim, mask, hood, neck_wrap,
  collar (+ rough lapel_l/r, coat, shoulder_l/r placeholders). Pilot objects built, shapes match source.

## To do, in order

1. Region seed files + builds by Sonnet subagents (brief: `docs/SEG_BRIEF.md`), one per region:
   chest+shoulders, arms, waist+hips, skirt+underskirt, legs+boots. Each writes
   `survey/seeds/<region>.json`, own labels file `survey/labels_<region>.npz`, builds its objects.
2. Lead: global `seeds grow`, fix leaks with extra seeds (known: hat_band creeping on crown front and
   brim top, small mask patch on brim front edge, fringe at collar/neck_wrap borders), rebuild all.
3. Studs/buttons as separate objects: mark stud faces from `survey/bumps.json` as cover of their parent
   (`--cover`, flattens them), place parametric studs from pass 2 (`scripts/hp_lead_studs.py`).
4. Assemble `hipoly3/` -> whole-model gate (target p95 < 1.5 mm, < 1% beyond 5 mm), comparison sheets
   per region vs source, export GLB/FBX (reuse `scripts/hp_assemble.py --export` logic), update docs.
5. Send the user the sheets + files. Commit and push to `retopo-plan` after every step.

## Defaults in force (user did not object)

Sculpt's pinched hat crown kept; no big belt buckle; no added boot buckles; reference-only items left
out; hip cord kept as its own object.
