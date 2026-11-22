# Review of the high-poly (pass 2) against the original model

Verdict: the user is right. The high-poly does not look like the source model or the references.
Side-by-side sheets: `docs/review_full.png`, `review_head.png`, `review_chest.png`,
`review_waist.png`, `review_arm.png`, `review_boots.png` (ours | source | source textured, same camera).

## Measured

Distance from the source surface to the nearest point of our model, in mm at 190 cm
(`scripts/review_deviation.py`, `docs/review_deviation.json`):

| Band | median | 95th pct | share of source more than 5 mm away |
|---|---|---|---|
| whole model | 2.3 | 15.7 | 21% |
| hips (y 0.47-0.62) | 4.4 | 26.3 | 47% |
| head + collar | 3.1 | 24.9 | 39% |
| boots | 2.8 | 14.6 | 26% |
| chest + arms | 2.2 | 16.5 | 22% |
| legs + skirt | 2.2 | 9.1 | 13% |
| belt | 2.0 | 8.0 | 11% |
| hat | 1.7 | 8.8 | 10% |

A faithful high-poly should sit under about 1 mm at the 95th percentile.

## What is wrong, by region

| Region | Source | Ours |
|---|---|---|
| Collar | tall cloth collar wrapping the back of the head, wide soft wings merging into the lapels | a vertical tube with two stiff fins |
| Mask | narrow tapered plate, pointed chin, long beak profile in side view | a square bucket visor |
| Hood / neck | head shape with ears set in, layered V neck wrap | a smooth cylinder; neck wrap is torn, with flipped faces |
| Hat | crown with two pinched dents, band with gold plate | dents removed on purpose (wrong call), band reduced to a thin ring |
| Coat upper | smooth tailored cloth, princess seams, armhole seams, back yoke, piped lapels | faceted stair-stepped surface, no seams, flipped faces at the armholes, lapels the wrong outline |
| Coat below the belt | the coat continues: overlap with 4 more buttons, pocket flaps | missing entirely; the skirt starts as a flat box under the belt |
| Belt | strap with studs, two keeper loops, small tab; no big buckle | an invented rectangular buckle, keepers wrong |
| Hip armour | per side: a broad upper band flap plus a large pointed shield with studs, front and back flaps | thin spikes floating inside the skirt; the most recognisable waist feature is gone |
| Skirt | panels flare from the waist with soft folds, piped diagonal front edges crossing at the centre | flat vertical boards with hard steps; the front crossing is two loose tubes |
| Sleeves | deep diagonal wrinkles | stair-stepped noise |
| Bracer / glove | close, but cuff band studs missing, glove surface noisy, pinky bent | (the best region) |
| Trousers | bloused over the boot top | straight tube with an invented band |
| Boots | plain shaft, one side tab, three overlapping instep bands, stitched toe cap, thin sole with heel | invented buckled ankle strap and second strap, triple-stacked sole, bands missing |

## Why it happened

1. **The gate measured topology, not likeness.** An object passed if it was all quads and manifold.
   Nothing failed a part for not resembling the source, and the summary reported quad counts as success.
2. **Shapes were re-invented instead of taken from the sculpt.** Collar, mask, hood, hip flaps, belt
   and boots were "modelled from parameters" by cheap builders reading a text spec. The sculpt already
   had these shapes correct; the builders' versions are guesses.
3. **Conform was too weak and the cages too coarse.** Projection used a short reach plus heavy
   smoothing on pass 1 cages built for a 25k-tri model, so cloth came out faceted and seams vanished.
4. **Wrong authority order.** I let the character sheet override the sculpt (buckle, second boot
   strap, hat dents) even though PLAN.md says the sheet is detail guidance only.
5. **No object for whole areas.** The coat below the belt and the upper hip bands had no register row
   that produced geometry, and the coverage check in OBJECT_PLAN step 2 was never run.
6. **The lead accepted work from per-part renders** and did not put the assembled model next to the
   source before delivering.
