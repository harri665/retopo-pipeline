# Part builder brief (read fully before starting)

You are rebuilding parts of a character as clean, game-ready LOW-POLY topology. Topology only: no UVs,
textures or materials. Another agent (the lead) reviews your output with scripts and renders, so only
measured results count.

## Setup
- Repo: `/home/user/retopo-pipeline`. Run everything from there.
- Python: `PY=/tmp/claude-0/-home-user/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/venv/bin/python`
- Run your script as a module: `$PY -m scripts.<name>` (not `python scripts/x.py`).
- Read `retopo/build.py`, `retopo/manifest.py`, `retopo/render.py` (short). Do NOT edit anything in
  `retopo/`; if you need a helper, put it in your own script. Do NOT run git commands.
- Source: a fused 2M-tri sculpt; you work against a 400k proxy via `B.Caster()` (BVH ray casts) and
  `meshio.load_proxy()` (V, F numpy arrays). It is ONE fused shell: there are no separate objects, and
  surfaces hidden under another item (sleeve under the bracer, coat under the belt) mostly do not exist.
- Units: figure height = 1.0. Axes: +Y up, +Z front, +X = the character's LEFT. Symmetry plane x = 0.46425.
- Reference images (the design authority): `reference/front.png`, `side.png`, `back.png`;
  `reference/character-sheet.png` is detail guidance only. View images with the Read tool.

## How to build
- One script per part: `scripts/<part>.py` writes `parts/<part>.obj` with `meshio.save_obj(path, V, polys)`.
- Look first: render close-ups of the source region (see `retopo/render.py`: `Cam(view, H, centre, half)`,
  `rasterize`, `shade`, `sheet`; views front/back/left/right/fl/fr/bl/br/top/bottom) and measure slices
  with numpy before choosing an axis and stations.
- Tubes (limbs, torso, boots, fingers, hat crown): `B.ring_cast` along an axis polyline, then
  `B.fill_nan`, `B.smooth`, `B.rings_to_points`, `B.grid_mesh(P, wrap_u=True)`. Mode `outer` gives the
  outermost surface; mode `all` with a `pick` function lets you choose among hits (for example ignore a
  strap, or take the surface under a flap). Where the rays hit a DIFFERENT item or miss, set the radius
  to NaN and let `fill_nan` continue the surrounding form. That is the intended hidden-surface fill.
- Sheets (skirt panels, flaps, collar, brim): sample a rows x cols grid of points on the surface
  (ray casts from a guide surface or along slices), then `B.grid_mesh(P)`. Thin free-standing sheets
  need thickness only where the inside can be seen; otherwise single-sided is right (say which you chose).
- Rigid bits (buckle, pauldron plates): simple boxes/bevels or a cast grid. Rivets, buttons and stitching
  are NOT modelled.
- Left/right: build the left, then `B.mirror_x`. Gate the mirrored part too; if it fails deviation
  because the source is asymmetric there, re-run your recipe on the right side instead.

## Topology rules
- All quads where the surface bends; triangles only on flat rigid caps. No n-gons.
- Even, roughly square quads; edge loops run around the limb and along it. No long thin slivers.
- Extra loops at joints: at least 3 rings at elbow, knee, wrist, ankle, shoulder; each finger joint 3 rings.
- Spend triangles on silhouette and bending, not on flat areas. Stay inside the triangle budget.
- Normals must face outward: in gate renders a face seen from behind is tinted RED. No red on the outside.
- Open ends that tuck inside another item (sleeve into bracer, trouser into boot, neck into collar) stay
  open and extend about 0.01 inside the covering item. Do not cap them.
- Each part is its own closed-or-open manifold shell; parts are separate objects and may overlap slightly
  where one sits on another (bracer over sleeve). No gaps visible from outside.

## Gate (run it yourself, iterate until it passes)
`$PY -m retopo.gate parts/<part>.obj --budget <tris> [--views front fl left bl back br right fr top bottom]`
- prints JSON metrics and writes `out/gate_<part>.png` (your wireframe; orange = source silhouette not
  covered by your mesh within the part's box) and `out/gate_<part>_src.png` (source, same cameras).
  LOOK at both with the Read tool every iteration.
- Pass criteria: `in_budget` true; `nonmanifold_edges` 0; `winding_errors` 0; `zero_area` 0;
  `sliver_ratio` < 0.02; `unused_verts` 0; `deviation.new_to_src.p95` < 0.004 (larger only where you
  deliberately invented a hidden surface, and you must list those places); renders show the right shape
  with no red outside faces, no holes, no spikes, no twisting.
- Orange in the render that belongs to a NEIGHBOURING item is expected; orange that belongs to your
  part is a miss to fix.

## Report back (final message, plain text)
For each part: file path, tris, the gate numbers above, single- or double-sided, what you invented
(hidden surfaces) and where, anything that does not match the references, and anything you could not
get to pass. Do not claim a pass you did not measure. Maximum three serious attempts per part; if it
still fails, say exactly how.
