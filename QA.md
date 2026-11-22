# QA: Infantry Ranged Trooper, topology pass 1

Topology only (no UVs, textures, materials, skeleton). Built from `models/scene.glb` (2M-tri fused sculpt).

## Deliverables
- `export/trooper_LOD0.glb|fbx`: 25,086 tris, 21 separate objects (`SM_Trooper_<part>`), 190 cm, feet at origin.
- `export/trooper_LOD1` (12,538 tris) and `LOD2` (6,264 tris): automatic decimation of LOD0, not hand-tuned.
- `parts/*.obj`: each part as quads in source units; `scripts/` rebuilds every part; `python -m retopo.assemble --export`
  rebuilds the checks and exports. `docs/assembly.png` shows wireframe vs source.

## Whole-model checks (measured)
- Non-manifold edges, winding errors, zero-area faces, unused vertices: 0 in every part.
- Silhouette overlap with the source, 8 angles: 94.1% to 95.1%.
- New surface to source: mean 0.0043, p95 0.012 (figure height = 1). The high tail is the deliberately invented hidden
  surfaces and inner walls of double-sided sheets.
- Source to new surface: mean 0.0043, p95 0.016; 12.7% of the source is more than 0.01 away. That is mostly detail left
  for a normal map (studs, buttons, straps, folds) plus the simplified areas listed below.

## Part status
| Part | Tris | Status |
|---|---|---|
| hat | 1120 | Good. Crown rebuilt without the two socket dents; band studs and emblem not modelled. |
| head (mask, wrap, neck) | 1240 | Good in the visible band. Ears are simple bumps; neck inside the collar is invented. |
| collar | 978 | Weakest part. Back and sides fine; front wings are chunkier than the source. |
| coat_upper | 1696 | Good. Smooth base; shoulders under the pauldrons and the neck are invented. Welds to the sleeves on a shared 16-point ring. |
| lapels | 376 | Usable. Two overlay sheets traced from the front view; 4.8% slivers near the pointed tip. |
| sleeve l/r | 480 each | Good. Shoulder under pauldron and end inside bracer are invented. |
| pauldron l/r | 672 each | Usable. Strap and plate as two shells; plate's pointed corners and rim not modelled. |
| bracer l/r | 640 each | Good. |
| glove l/r | 1212 each | Usable. All quads, proper knuckle loops; palm around the thumb is a little uneven. |
| belt, buckle, hip_flaps | 768 / 216 / 1280 | Good. Buckle is a simple block. |
| skirt | 4476 | Usable. Three double-sided panels, even grid for cloth; hem is rounded where the source has pointed corners; pleats simplified. |
| underskirt | 1256 | Usable. Tattered inner layer front and back; side tatters missing; rows zigzag at the hem. |
| trousers | 2288 | Good below the hips; crotch and hips (hidden under the coat) are invented. |
| boot l/r | 1692 each | Usable. Shaft under the trouser is invented; sole is a flat slab; straps are silhouette bumps only. |

## Decisions taken without input (change if wrong)
- Budget target was about 40k tris; LOD0 landed at 25k, so there is room to add detail where it is weak.
- Chest-to-hip cord removed. Rivets, buttons, studs, stitching, buckles left for the normal map.
- Left side built and mirrored for arms, boots, hip flaps and collar; small source asymmetries are lost.
- Right/left arm, glove and boot are separate objects; coat_upper and sleeves share exact seam vertices but are not welded.

## Not done
- UVs, bake, textures, materials (out of scope for now).
- Deformation test with a skeleton.
- Hand-tuned LODs.
