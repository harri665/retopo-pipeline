# High-poly model: Infantry Ranged Trooper

Topology only (no UVs, textures or materials). Built object by object from `docs/REGISTER.md`.

## Deliverables
- `export/trooper_highpoly.glb` and `.fbx`: 124 named objects (`HP_Trooper_<object>`), 754200 quads, 190 cm tall.
- `hp_cages/<object>.obj`: the quad control cage of each built object (the subdivision topology).
- `hipoly/<object>.obj`: each object subdivided (and conformed to the sculpt where it is cloth or leather).
- `python -m scripts.hp_assemble --export` seats cloth under hardware, checks every object and re-exports.
- `docs/hp_assembly.png` (colour per object beside the sculpt), `docs/hp_report.json` (per-object numbers).

## Measured
- 124 of 124 objects are 100% quads with 0 non-manifold edges, 0 winding errors, 0 zero-area faces.
- 23 objects have open borders; each is an opening buried in a neighbour (sleeve ends, glove wrist, boot top, neck).
- `mask_faceplate` carries 7 unused vertices (harmless; not yet cleaned).
- Not measured: object-to-object intersection across the whole model (checked per group in renders only),
  and deformation (no skeleton).

## How each kind of object was made
- Modelled clean from measurements: hat band, staples and emblem, mask faceplate, slits and rim, all studs and
  buttons, belt strap, end blocks, tab and buckle, bracer cuff, straps, boot sole, heel block, welt, straps, buckles.
- Modelled shape seated on the sculpt: hat body, collar, lapels, shoulder strap / cap plate / flap, bracer tube and
  dorsal plate, hip flaps and shields, boot toe cap and heel counter, skirt panels.
- Quad cage conformed to the sculpt's folds: coat body, sleeves, trousers, hood, scarf, gloves, boot upper,
  underskirt strips, liners.
- Swept trims: lapel, closure, collar, mask, shoulder, hip flap and skirt piping.

## Known weak spots
- Coat body: still the first-wave version (pass 1 cage conformed). The closure overlap edge is only piping,
  shoulders under the armour are invented, and a gap can remain under the arch strap at the armhole.
- Collar: wings stick out more sideways than the sculpt's upswept wings; piping stops 2 cm short of the lapel corner.
- Mask: three slits modelled; the reference's two short outer ticks are not.
- Skirt back panel: pleats are subtle; the hem's raised double-line border is not modelled. Underskirt has 16
  strips (sculpt has more, some hidden); thin strip tips are blunt.
- Boots: upper is smooth (soft folds above the ankle strap missing); toe spring not modelled.
- Glove: slight kink at the base of the little finger.
- Studs: positions come from automatic detection, mirrored for symmetry; a few are missing (for example one of
  four on each shoulder cap) and none have been re-seated on the final surfaces.
- Several objects exceed the 15k-quad guideline (skirt back panel 32k, hip flap trims 22k).

## Built without a decision from the user (see register)
No hair; no back cross emblem or centre trim; no shoulder emblems; no skirt filigree; no hip cord;
buckle placed on the belt tab although the references show none at the front.
