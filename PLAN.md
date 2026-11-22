# Retopology plan: Infantry Ranged Trooper

Scope: **topology only**. No UVs, bakes, textures or materials for now.
Status: topology pass 1 complete, see `QA.md`. Next candidates: collar wings, skirt hem, underskirt sides, deformation test.

## Inputs

- `models/scene.glb`: 2,000,000 tris, 1,018,355 verts (999,982 welded), ONE fused watertight
  shell (plus 3 stray fragments of 16/6/2 tris to delete). No boundary or non-manifold edges,
  7 degenerate tris. One material, no skeleton, no normal map. Positions are uint16 with node
  scale 1.52476132e-05; UVs float32.
- Axes: +Y up, front faces +Z, character's left is +X. Extent 0.9285 x 0.9993 x 0.2918,
  centre x = 0.464. Height is normalised to 1 unit; rescale at export (assume 190 cm with hat).
- `reference/front.png`, `side.png`, `back.png`: near-orthographic T-pose, match the model pose.
  These are the authority. `reference/character-sheet.png`: relaxed pose and close-ups, detail
  guidance only (it disagrees on pauldron design and hat emblem).

## What the mesh is like

- Sculpt-quality source: buttons, rivets, straps, folds are real geometry.
- Already free-standing thin shells with inner faces: coat skirt panels, collar, hat brim,
  tattered underskirt strips.
- Full legs exist under the coat; crotch arch at about y = 0.55.
- Torso from about y = 0.55 to 0.70 is one fused blob (coat = body).
- Rough heights: hat brim about 0.90, head 0.80-0.90, belt 0.623-0.651 (hip flaps 0.47-0.636), boot tops about 0.17.

## Defaults (user has not overridden)

- Target Unreal, about 40k tris for LOD0, plus LODs.
- Deliver a rig-ready mesh; no skeleton or skinning.
- Remove the chest-to-hip cord on one side (not in any reference).
- Skirt panels at moderate, even density (usable for cloth or bone chains).

## Shape fixes from references (geometry only)

- Hat crown has two eye-socket dents: smooth to match `front.png`.
- Hidden surfaces after separation (sleeve under pauldron and bracer, coat under belt and hip
  flaps, chest behind lapels, neck inside collar): continue the surrounding form.
- Deferred with textures: back ornament, shoulder emblems, visor glow.

## Parts

hat; mask + head wrap; collar; coat upper + sleeves; skirt panels (front L, front R, back pair);
tattered underskirt; pauldrons L/R; bracers L/R; gloves L/R; belt + buckle + hip flaps;
trousers; boots L/R. Rivets and small buttons are NOT modelled (normal map later).

## Method

1. **Toolkit** (lead): loader with npz cache, about 400k-tri proxy for fitting, orthographic
   renderer (clay, wireframe, part colours, source underlay), metrics script.
2. **Label the source** (lead): no islands or materials to use, so label by body-coordinate rules
   (limb axes, radius jumps along each axis for bracer/glove/boot/belt edges, thin-shell
   detection for skirt/collar/brim), then snap boundaries to creases. Check with colour renders.
3. **Rebuild each part, do not auto-remesh** (subagents in parallel):
   - limbs, torso, boots: quad tubes built by casting rings of rays from the limb axis onto that
     part's labelled source; extra rings at shoulder, elbow, knee, ankle; where rays miss or hit
     another part, interpolate the radius (this is the hidden-surface fill)
   - skirt panels: quad grids resampled from horizontal slices of each panel
   - hat: concentric rings; pauldrons, bracers, belt, buckle: rigid rebuild or decimate
   - gloves: finger tubes joined to a palm (hardest; most likely to need a manual pass)
   - build left side once and mirror where the source is symmetric enough
4. **Gate** (script, then lead): tri count in budget; deviation from source under tolerance;
   no non-manifold, zero-area or flipped faces; sliver ratio under limit; silhouette overlap
   from 8+ angles; wireframe renders reviewed against references.
5. **Assemble and export** (lead): scale, pivots, naming, LODs, FBX + GLB, QA report with
   overlay renders and a log of every invented surface.

## Subagent split

- Lead: toolkit, labelling, part manifest (axes, ring stations, budgets), every pass/fail call.
- Mid-tier model, one agent per group: arms; legs; skirt; head; torso and belt.
- Smallest model: LODs and render sheets.
- Max three attempts per part, then the lead takes it over.

## Environment notes

- Container is temporary: 4 CPU, 15 GB RAM, no GPU. Anything not pushed is lost.
- Working stack (reinstall if the venv is gone): Python 3.13 venv with
  `pip install bpy trimesh pymeshlab scipy pillow numpy xatlas`.
  Verified: bpy 5.3.0 (QuadriFlow operator and BVHTree present), pymeshlab filters load,
  numpy 2.5.5, scipy 1.18.1.
- `libopengl0` / `libegl1` are not available from apt, so no GL rendering: use Cycles CPU or a
  numpy rasteriser. pymeshlab prints plugin warnings about this but its filters work.
- The GLB can be parsed directly with numpy (header JSON + binary chunk); a 2x2 point splat of
  verts plus face centroids gives a usable preview of the 2M mesh in seconds.
