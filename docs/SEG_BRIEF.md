# Brief: segment a region of the source sculpt into objects and rebuild each object from its segment

You work in `/home/user/retopo-pipeline` (do NOT git commit/push; the lead does). Python:
`PY=/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python`, always run from the repo root as modules.

Goal: the rebuilt objects must look like the SOURCE sculpt (`models/scene.glb`). You do not design or
invent shapes. You only (1) tell the tool which part of the sculpt is which object, (2) run the builder,
(3) check likeness. Geometry only, no materials.

Model facts: +Y up (height 0..1), front is +Z, character's LEFT is +X, symmetry plane x = 0.46425.
`docs/REGISTER.md` lists the objects expected per region (names there are guidance; trust what you see in
the sculpt, and do NOT add things the sculpt lacks).

## Step 1: look at your region

`$PY -m retopo.survey <tag> cx cy cz half front,left,back,right` writes `out/survey_<tag>_clay.png` and
`_tex.png` with coordinate ticks (y on the left edge; x or z along the bottom). Read them (Read tool).
Use several zoom levels. Crop big sheets with `$PY scripts/_crop.py in.png out.png x0 y0 x1 y1` (fractions).

## Step 2: write seeds -> labels

Create `survey/seeds/<region>.json`:

    {"objects": {"sleeve_l": [["front", 0.70, 0.76], ["back", 0.70, 0.76], ["top", 0.70, 0.16]], ...},
     "mirror": ["sleeve_l"]}

A seed `[view, a, b]` (optional 4th value radius, default 0.004) marks the first surface seen from that
view at those coordinates: front/back use (x, y); left/right use (z, y); top/bottom use (x, z).
`"left"` looks at the character's left side (from +X), `"right"` from -X. A view like `"bottom>0.9"` or
`"top<0.6"` only considers surface with y above/below that value (to reach under or inside things).
Labels grow from seeds over the surface and stop at creases, so: several seeds per object spread over
it (front, back, top, sides), and extra seeds near any border that ends up wrong. Names ending `_l` are
the character's left (+x); list them under "mirror" to copy seeds to `_r` (check the right side too: the
sculpt is not perfectly symmetric; add `_r` seeds by hand where needed).
Every surface in your region must belong to some object. Seed neighbouring regions' objects too, using
the shared names below, so borders are contested (duplicate seeds across files are fine).

Grow and view (3 s + render):

    $PY -m retopo.seeds grow --out survey/labels_<region>.npz
    $PY -m retopo.seeds view out/seg_<region>.png cx cy cz half front,fl,left,back,right,top --labels survey/labels_<region>.npz

"WARN ... seed misses the surface" means that seed hit nothing: fix its coordinates. Iterate until each
object's colour covers exactly that object in every view, with borders on the real seams. Other agents
edit their own seed files at the same time; only edit yours.

## Step 3: build and check

    $PY -m retopo.fromseg <name> --edge 3 --labels survey/labels_<region>.npz

Options: `--edge` base quad size in mm before one subdivision (2 to 2.5 for small or detailed objects,
3 to 4 for large cloth); `--sharp` for hard-edged plates; `--levels 1` (default). It prints JSON with
`PASS`. Required: seg_to_obj_p95 <= 1.0 mm, obj_to_seg_p95 <= 1.0 mm, quad_ratio 1.0, nonmanifold 0,
winding 0. If "quadriflow failed", the segment is usually fragmented: fix the labels (or try another --edge).
Then look, new beside source:

    $PY -m scripts.review3 out/hp3_<region>.png cx cy cz half front,fl,left,back hipoly3 name1 name2 ...

Passing numbers are not enough: in the picture every object must be recognisably the same thing as in
the source, with no missing pieces, no stray strips from neighbours, no big holes between objects.
Fix by improving seeds, rebuild, look again. Put all your build commands in `scripts/hp3_<region>.sh`.

## Report back (short)

Object list with PASS and the two p95 numbers each, path of your final review image, and an honest list
of what still differs from the source or which borders are still wrong. No long narration.

## Shared object names (use exactly these; add more in your own region as the sculpt requires)

hat_crown hat_band hat_brim mask hood neck_wrap collar | coat_upper lapel_l lapel_r
shoulder_arch_l/r shoulder_front_strap_l/r shoulder_cap_l/r shoulder_flap_l/r | sleeve_l/r bracer_cuff_l/r
bracer_tube_l/r bracer_plate_l/r wrist_strap_l/r glove_l/r | belt belt_tab coat_lower pocket_flap_l/r
hip_band_l/r hip_shield_l/r hip_cord | skirt_front_l/r skirt_side_l/r skirt_back skirt_liner
underskirt (or underskirt_NN per strip) | trousers boot_shaft_l/r boot_foot_l/r boot_sole_l/r
