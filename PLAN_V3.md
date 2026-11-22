# Revised plan (pass 3): likeness first, source-driven

Scope unchanged: geometry only, clean quad high-poly split into individual objects.
See `REVIEW.md` for what went wrong in pass 2. Passes 1 and 2 stay in the repo as tooling; their
shapes are not reused except where listed under "Keep".

## Principles

1. **The sculpt is the shape.** Every object's form comes from the source surface. References are
   used only to decide where one object ends and the next begins, and to fill surface the sculpt
   does not have (hidden areas). Nothing visible in the sculpt is redesigned.
2. **Likeness is a hard gate, checked before topology.** No object and no assembly is accepted on
   quad counts alone.
3. **Invented geometry is opt-in.** Anything not in the sculpt (big buckle, second boot strap,
   back emblem, filigree) is left out unless the user asks for it.

## Step 1: Segment the source itself

Label every one of the 2M source faces with an object id, instead of building objects elsewhere and
hoping they cover it.

- Seeds: texture colour clusters (gold / leather / cloth), crease lines (dihedral angle), thickness
  steps, plus the positions already measured in `docs/REGISTER.md`.
- Grow regions on the mesh graph and stop at creases; the lead fixes borders with painted
  plane/box cuts where automatic borders are wrong.
- Output `survey/labels.npy` and a colour-per-object render from 8 views.
- **Gate:** no unlabelled faces; every register row has faces; lead compares the label render to
  the textured source. This is the coverage check pass 2 skipped.
- New rows the review found missing: coat lower (overlap + 4 buttons), pocket flaps, upper hip band
  flaps (front and back, both sides), pointed hip shields, instep bands x3, boot side tab.

## Step 2: Per object, take the shape from its segment

| Kind | Objects | How |
|---|---|---|
| Cloth and organic | coat upper and lower, collar, lapels, sleeves, neck wrap, hood, ears, skirt panels, underskirt strips, trousers, gloves, boot upper | Cut the segment out, close the hidden side by continuing the surface under its neighbours, quad-remesh that surface directly with flow guided by its borders and creases (QuadriFlow / field-aligned), then multi-level subdivide and project with the full reach so folds and seams survive. Shells get thickness after projection. |
| Rigid plates fitted to the body | mask, shoulder plates, bracer plates, hip bands and shields, hat crown/brim/band, belt, boot sole and toe cap | Same as above from the segment, then clean only: straighten rims, sharpen edges with support loops, make left and right symmetric. The outline and proportions stay the sculpt's. |
| Small repeated hardware | studs, buttons, keeper loops | Keep pass 2's parametric parts (they matched), but re-detect per labelled segment so none are missing, and seat each on its final parent surface. |

Removed from the build: belt buckle frame, boot ankle buckle strap and second strap, extra trouser
bands, smoothed hat crown (the sculpt's pinched crown comes back).

## Step 3: Likeness gate (new, per object and for the whole model)

An object fails, whatever its topology, unless all of these hold:

- source segment to object distance: 95th percentile under 1.0 mm for cloth and plates, max under
  3 mm (hidden-side fill excluded);
- object to source distance: 95th percentile under 1.5 mm (catches invented geometry);
- silhouette overlap with the segment over 98% from 8 views;
- normal-difference render against the source shows no faceting or stair steps;
- side-by-side sheet (ours | source | textured source) looked at by the lead and by a reviewer
  subagent that did not build it, asked to list visible differences.

Then the existing topology gate: 100% quads, no non-manifold edges, no flipped or zero-area faces.
Whole-model targets, from the pass 2 baseline of 15.7 mm p95 and 21% beyond 5 mm: p95 under 1.5 mm,
under 1% beyond 5 mm, with every region sheet reviewed against the source before anything is sent.

## Step 4: Order of work

1. Segmentation and label render (lead). Stop and show the user the label map.
2. One pilot region end to end, the worst one: head + collar. Show it beside the source. Only if
   it passes the likeness gate does the method go wide.
3. Waist and hips, then coat, then skirt, boots, arms, hat.
4. Assemble, run the whole-model gate, export, deliver with the comparison sheets.

## Subagents

- Builders (mid-tier): one region each, but they now receive the labelled segment meshes and the
  likeness script, and must report the gate numbers; they do not design shapes.
- Reviewers (small model, never the builder): difference lists from side-by-side sheets.
- Lead: segmentation borders, hidden-surface decisions, every accept/reject, and the assembled
  comparison before delivery.

## Keep from pass 2

Tooling (`retopo/`), stud and button parts and detection, bracer and glove as a starting point,
sub-division and crease code, tuck-under, export.

## Decisions for the user (defaults in brackets)

1. Hat crown: keep the sculpt's two pinched dents [yes].
2. Belt: sculpt's plain strap with tab and keepers, no big buckle [yes].
3. Boots: sculpt's single tab and instep bands, no added buckles [yes].
4. Items only in the references (back cross, shoulder emblems, skirt filigree, nape hair) [leave out].
5. Hip cord on the right side, only in the sculpt [keep, as its own object].
