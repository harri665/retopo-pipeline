# High-poly v3 (pass 3): rebuilt from the source sculpt's own surface

Replaces the pass 2 high-poly (`HIGHPOLY.md`), which failed on likeness (`REVIEW.md`).
Method: `PLAN_V3.md`. Files: `export/trooper_highpoly_v3.glb|fbx`, objects in `hipoly3/`.

## Result

| | pass 2 | v3 |
|---|---|---|
| source surface to model, 95th percentile | 15.7 mm | 0.22 mm |
| source surface more than 5 mm from model | 21% | 0.49% |
| source surface more than 1 mm from model | not measured | 1.9% |
| objects | 124 | 87 (62 segments + 25 stud/button sets, 76 studs and buttons) |
| quads | 754,200 | 1,866,768 |
| all quads / non-manifold edges / flipped faces | yes / 0 / 0 | yes / 0 / 0 |

Per band (95th pct mm, share beyond 5 mm): hat 0.06 / 0%; head+collar 0.34 / 0.47%; chest+arms 0.17 / 0.03%;
belt 1.29 / 0.65%; hips 0.35 / 0.98%; legs+skirt 0.21 / 0.50%; boots 0.18 / 0.91%. Numbers: `docs/hp3_assembly.json`.
Side-by-side sheets (ours | source): `docs/hp3_full.png`, `hp3_head.png`, `hp3_chest.png`, `hp3_arm.png`,
`hp3_waist.png`, `hp3_skirt.png`, `hp3_boots.png`.

## How it was made

1. Every face of the 2M sculpt is labelled with an object (`survey/seeds/*.json` -> `python -m retopo.seeds grow`
   -> `survey/labels.npz`). Labels grow from hand-placed seeds and stop at creases; stray islands are absorbed.
2. Each object is remeshed from its own segment (`retopo/fromseg.py`): QuadriFlow quads, one subdivision,
   projection back onto the segment. Borders are extended a few millimetres and sunk so neighbours overlap.
3. Detected studs and buttons are cut out of their parent, the parent is rebuilt smooth underneath, and
   clean parametric studs/buttons are seated at the measured positions (`scripts/hp3_studs.py`).
4. `scripts/hp3_build_all.py` builds everything; `scripts/hp3_assemble.py --sheets --export` checks and exports.

## Known limits

- Topology is automatic quad flow: all quads, but with scattered poles, not the hand-planned edge loops
  of the pass 1 low-poly. Good for sculpt/bake source, not for deformation as is.
- 1.87M quads is heavy; density can be lowered per object with `--edge`.
- Objects are the visible surface of each item with tucked borders, not closed solids; surfaces hidden
  under other objects are not continued.
- Seams that do not sit on a real crease show as a shallow, slightly ragged groove: coat/right-lapel
  closure edge, back armhole seams, back ends of the hip bands.
- Small holes remain (the 0.49%): mostly in the front-right skirt panel and under the belt at the back.
- Only detected hardware is separate (76 pieces). Undetected studs and some coat buttons are still bumps
  on their parent.
- Trousers and underskirt are single objects made of 5 pieces each (underskirt strips are not split).
- Three parents with separated studs (belt_tab, shoulder_arch_l/r) fail the per-object "invented shape"
  number only because that check still compares against the sculpt with studs on.
