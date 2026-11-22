# Object-awareness plan (pass 2)

Pass 1 produced clean topology but treated the character as about 14 big regions. It never asked
"what individual things is this character wearing?", so buckles became blocks, armour plates became
curved sheets, and studs, straps, buttons and ears were dropped or smeared. Pass 2 puts an explicit
object survey and a build-method decision in front of any geometry work.

## Step 1: Survey every object (before building anything)

Go region by region, top to bottom: hat, head, neck, collar, chest, shoulders, upper arm, forearm,
hand, waist, hips, skirt, legs, feet. For each region collect four kinds of evidence:

1. **References.** Crop the matching area from `front/side/back.png` and the close-ups on
   `character-sheet.png`. The sheet's detail call-outs are a ready-made list of what the designer
   considers separate items.
2. **Source close-ups.** Clay renders of the 2M source (not the proxy) from at least 6 angles per
   region at high zoom, plus a curvature/crease render so seams and edges stand out.
3. **Texture.** Pass 1 ignored the base-colour map. Project it onto the mesh and cluster by colour:
   gold trim, dark leather, blue cloth and metal separate cleanly and mark object borders that the
   geometry alone does not show.
4. **Layering.** Cross-sections through the region. A thickness step or a free edge means one thing
   sits on top of another, so it is its own object.

Each region is surveyed twice, by two cheap subagents that do not see each other's list. The lead
merges the lists and looks again wherever they disagree. A count mismatch (one says 3 boot straps,
the other 2) is a prompt to look, not something to average.

## Step 2: Write the object register

One row per object, in a tree (boot > upper, sole, heel, strap x3, buckle x3). Each row records:

- name, parent, how many instances, mirrored or unique
- material (cloth, leather, metal) and whether it is rigid or bends
- where its border is, and the evidence images that show it
- how it attaches (sits on top, tucks under, welded seam)
- whether it must move on its own when rigged (hat, belt pouch) or rides with its parent
- whether the source shape is trustworthy or lumpy AI output
- whether references and mesh disagree, and which wins
- build method and geometry level (step 3), triangle budget

The register is finished only when a coverage check passes: every part of the source surface is
assigned to exactly one register row, shown as a colour-per-object render with no grey (unassigned)
areas, and every item called out on the reference sheet has a row.

## Step 3: Decide how each object is made

| Method | Use when | Typical objects |
|---|---|---|
| **Model from scratch**, using the mesh only for position and size | Rigid, manufactured, geometric; or repeated; or the sculpt version is lumpy | buckles, buttons, studs and rivets, belt, hat band, emblem, boot sole and heel, mask faceplate, strap keepers |
| **Retopologise from the mesh** | Soft or organic, with folds unique to this sculpt | coat body, sleeves, trousers, skirt panels, underskirt strips, gloves, scarf, hood, ears |
| **Hybrid**: model a clean shape, then conform it to the mesh | Rigid but curved to fit the body | pauldron plates, bracers, boot upper, collar, lapels, hat crown and brim, hip flaps |
| **No geometry**, normal map or texture later | Too small to change the silhouette | stitching, fine wrinkles, sole tread, engraved lines |

Rules that settle the unclear cases:

- If it sticks out more than about 3 mm at game scale or breaks the outline, it gets geometry at
  LOD0. Smaller than that goes to the normal map.
- If it repeats (studs, buttons, straps), model one and place copies at positions measured from the
  mesh. Never rebuild each copy from the sculpt.
- If it is rigid and symmetric, model it clean and ignore the sculpt's wobble.
- If the reference shows it and the mesh does not (back ornament, shoulder emblems), model it from
  the reference and log it as invented.
- If it only exists in the mesh and no reference shows it (the chest cord), ask before keeping it.

## Step 4: Spec sheet per object, then build

Before a builder starts, the lead writes a short spec: reference crops, source close-ups, measured
size and position, method, budget, how it attaches, and what "matches" means for this object.
One object (or one repeated set) per builder task. Small modelled items are built parametrically
(for example `buckle(width, height, thickness, bevel)`) so they are easy to correct.

## Step 5: Check each object against the references

Pass 1's gate measured distance to the sculpt, which says nothing about a modelled buckle. Add:

- a per-object close-up, new model beside the source and the reference crop, reviewed by a subagent
  that did not build it and is asked "is this the same object?"
- instance counts and positions checked against the register (10 buttons expected, 10 present)
- the existing topology checks, and silhouette overlap per region

## Preliminary inventory (from pass 1 renders; step 1 will confirm or correct it)

| Region | Objects seen so far | Likely method |
|---|---|---|
| Hat | crown, brim, band, band studs, front emblem, cord loops | hybrid; band, studs, emblem modelled |
| Head | mask faceplate, visor slits, mask rim, hood/wrap, ears x2, hair (none visible yet: check) | mask modelled; hood and ears retopo |
| Neck | scarf/wrap, inner collar layer | retopo |
| Collar | band, wings x2 | hybrid |
| Chest | coat body, lapels x2, closure flap edge, buttons (about 10), trim lines | body retopo; lapels hybrid; buttons modelled |
| Shoulders | strap plate, arm plate, raised rims, studs (about 6 per side) | plates hybrid; studs modelled |
| Arms | sleeve, bracer body, end bands, strap and tab, studs, glove, glove cuff seam | sleeve and glove retopo; bracer hybrid |
| Waist | belt strap, buckle frame, tongue, keeper plates, belt studs | all modelled |
| Hips | upper flap, lower pointed flap, flap studs, trim, cord | hybrid; studs modelled |
| Skirt | front panels x2, side pleat panels, back panel, hem trim, inner liner | retopo |
| Underskirt | individual tattered strips | retopo per strip |
| Legs | trousers, cuff bands | retopo |
| Feet | boot upper, toe cap, sole, heel, straps (2 to 3), strap buckles | upper hybrid; sole, heel, buckles modelled |

## Subagent use

- Surveyors (smallest model, two per region, independent): list objects with evidence crops.
- Lead: merges the register, makes every method decision, writes specs, accepts or rejects.
- Builders (mid-tier): one object or repeated set each.
- Reviewers (smallest model, never the builder): per-object "same object?" check against references.

## Open questions for the user

1. Studs, rivets and buttons: real geometry at LOD0, or normal map only?
2. Which items must move on their own when rigged (hat, hip flaps, belt, straps)?
3. Keep or drop the chest-to-hip cord?
4. Triangle ceiling for LOD0 once the small objects are added (pass 1 used 25k of about 40k).
