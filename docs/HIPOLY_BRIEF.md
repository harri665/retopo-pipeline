# High-poly object builder brief (read fully before starting)

Goal: a clean HIGH-POLY model of the character, made of separate, properly named objects, each with
proper subdivision topology. You build the objects of ONE group from `docs/REGISTER.md`.
The lead reviews renders and metrics; only measured, looked-at results count.

## Setup
- Repo `/home/user/retopo-pipeline`; run from there.
  `PY=/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python`
  Run scripts as modules: `$PY -m scripts.<name>`. Do not edit `retopo/`, other groups' files, or run git.
- Units: figure height = 1.0. +Y up, +Z front, +X = the character's LEFT. Symmetry plane x = 0.46425.
- Read: `docs/REGISTER.md` (your group's table), then skim `retopo/build.py` (ray casting, grids, tubes,
  solidify, mirror), `retopo/hard.py` (studs, buttons, sweep, sections, rounded paths, boxes),
  `retopo/hipoly.py` (subdivide with creases, project onto the sculpt), `retopo/view.py`.
- Evidence: `out/survey_<region>_tex.png` (painted sculpt, 8 angles, coordinate ticks) and `_clay.png`;
  `reference/front.png|side.png|back.png` (authority) and `reference/character-sheet.png`.
  Make more close-ups whenever you need them: `from retopo.survey import sheets; sheets(name, [cx,cy,cz], half, views)`
  writes `out/survey_<name>_tex.png` and `_clay.png` (prefix your names with your group letter).
- The sculpt is one fused, lumpy AI-generated shell. Use it for position, size and soft folds. Do NOT copy
  its wobble onto manufactured things.
- `survey/bumps.json` lists detected studs/buttons (centre, normal, radius, height). The LEAD places all
  studs, rivets and buttons from it. Do not model those; do keep them from denting your surfaces
  (pass `exclude=[(centre, radius*1.6+0.002), ...]` to `hipoly.project`).

## What to deliver per object
1. `scripts/hp_<group>_<object>.py` that writes
2. `hp_cages/<object>.obj`: the quad control cage (this is the "proper topology"), and
3. `hipoly/<object>.obj`: the cage subdivided 2 levels (`hipoly.subdivide(V, polys, 2, crease_angle=..)`,
   or explicit support loops), and for soft objects conformed (`hipoly.project`).
Left/right pairs: build the left (+X), `B.mirror_x` for the right, unless the survey says they differ.
Object names come from the register (`shoulder_cap_plate_l`, `belt_band`, ...). One object per file;
repeated identical items may share a file (`boot_buckles_l`).
When your replacement objects are done, delete the superseded first-pass files listed in your task.

## Method per register row
- **model**: build from parameters with `retopo/hard.py` or your own code. Measure position, size and
  orientation on the sculpt (ray casts, slices, the tick marks). Straight edges straight, circles round,
  symmetric things symmetric, constant thickness. Hard edges get a crease or a pair of support loops so
  they stay crisp but slightly rounded after subdivision (never razor sharp, never blobby).
- **conform**: quad cage following the form (loops around limbs, along edges and seams), subdivide,
  then `hipoly.project` so it takes the sculpt's folds. Check it did not grab a neighbouring object.
- **hybrid**: model a clean shape (plate, strap, flap, cap) with real thickness, then seat it on the
  sculpt: fit its guide curve/outline to the sculpt, or project only its outer face and re-thicken.
- **trim**: `hard.sweep` a small section (`rect_section` or `round_section`) along the edge of the
  parent object (take the path from the parent cage's border vertices, normals from the parent).

## Topology rules (high-poly)
- 100% quads (a triangle only where a modelled piece ends in a real point; say where).
- Even quad size within an object; loops follow the object's edges and curvature.
- Every hard edge is supported (crease or two loops). No pinching, no unsupported corners rounding off.
- Thin things have real thickness (solidify) with a clean rim. No zero-thickness sheets.
- Each object is one clean manifold shell (closed, or open only where the opening is buried inside
  another object). No non-manifold edges, no flipped faces (red in renders = facing away), no
  self-intersection. Objects overlap/sink into each other slightly where they touch; no visible gaps.
- Keep it reasonable: a typical object is 1k-15k quads after subdivision.

## Check every object (iterate until right)
- `$PY -m retopo.gate hipoly/<object>.obj --views front fl left bl back br right fr` for the topology
  numbers (ignore `budget`; deviation only matters for conform objects).
- Side-by-side with the sculpt: `$PY -m retopo.view out/<group>_<name>.png cx cy cz half view1,view2,.. hipoly/<obj1> hipoly/<obj2> ... --nowire`
  (drop `--nowire` to see the wireframe; pass `hp_cages/<obj>` to see the cage). LOOK at the result
  with the Read tool next to the `_tex` survey sheet and the reference crop: does it read as the same
  object, same size, same place, same number of parts?
- Also render your whole group together with its neighbours (existing files in `hipoly/`) to check seating.

## Report (final message)
A table: object, file, quads, method used, gate numbers (nonmanifold, winding_errors, zero_area,
quad_ratio), "matches sculpt/reference: yes / partly / no" with one line why. Then: anything in the
register you found to be wrong (counts, positions, existence), anything you invented, what is still weak.
Do not claim a match you did not look at.
