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

## Status update

- Five Sonnet region builders launched (chest, arms, waist, skirt, legs), each following
  `docs/SEG_BRIEF.md`; they write `survey/seeds/<region>.json`, `survey/labels_<region>.npz`,
  `hipoly3/<object>.obj`, `scripts/hp3_<region>.sh`. If they are gone after a reset, check which seed
  files exist and relaunch only the missing regions with the same prompts (region descriptions are in
  the brief + REGISTER). Then continue with to-do item 2.
- arms DONE (14 objects, all PASS; added bracer_ring_strap_l/r). Open: coat_upper seeds in chest.json at
  left/right z 0.10-0.19, y 0.70-0.75 land on the arm (fix in global pass); ragged cuff/sleeve border;
  sleeve upper end vs shoulder flap untuned; right arm not visually checked.
- chest DONE (11 objects PASS: coat_upper, lapel_l/r, shoulder_arch/cap/flap_l/r, armhole_roll_l/r).
  Open: ragged coat_upper hem at belt; lapel_r lower end may be long; armhole roll borders soft;
  arms.json shoulder_flap_l seeds at front (0.62,0.79), top (0.62,0.16), back (0.64,0.78) are really on
  the cap (chest.json overrides them only by file order: fix in arms.json during the global pass).
- waist DONE (10 objects PASS: belt, belt_tab, coat_lower, hip_band_l/r, hip_shield_l/r, pocket_flap_l/r,
  hip_cord at --edge 2). Open (lead must fix in global pass): hip shields cut short at the pointed tip
  (skirt seeds placed just outside the outline stole it; right shield sits ~0.045 further back in z);
  coat_lower solid only in the upper front, breaks into strips lower down; shield/skirt border is a
  weak crease -> needs dense seeds on both sides.
- legs DONE (15 objects PASS). Boots good: boot_shaft/tab/band1/band2/toe/foot/sole _l/_r.
  trousers WRONG: contains skirt/underskirt panels (x 0.33-0.37 and 0.52-0.58 from y 0.2 up), upper
  legs barely captured, label fragmented; its build takes ~30 min at edge 3.5 (too dense/slow: after
  fixing labels use --edge 5). Fix needs skirt seeds. Also remove chest.json sleeve_l seed
  ["bottom", 0.6, 0.16] (lands on the left boot sole).
- skirt DONE by builder (7 objects PASS but panels moth-eaten: holes from stray seeds in legs.json /
  waist.json, side/back border arbitrary). ALL FIVE BUILDERS FINISHED; no agents running.
- Lead changes since: `seeds grow` now auto-absorbs stray label islands (`segment.absorb_islands`);
  `fromseg.build` extends each object by a sunk margin (margin_rings=4, sink_mm=1) so seams overlap in a
  groove instead of cracking (verified on lapel/coat); `scripts/hp3_build_all.py [names]` builds all
  labels 4 at a time with per-name edge sizes -> `docs/hp3_report.json`.
- Upper body (32 objects) rebuilt from global labels and looks like the source (`out/hp3_upper.png`);
  needs one more rebuild with margins. armhole_roll_r failed (3.7 mm): check its label.
- NEXT: review merged lower-body labels (`seeds view`), fix trousers / hip shields / coat_lower / skirt
  holes with seeds, then `python -m scripts.hp3_build_all` (everything), then to-do items 3-5.
