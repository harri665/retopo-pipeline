# Object survey brief

You are surveying ONE region of a character sculpt to list every distinct physical object in it.
You only look and write a list. Do not build geometry, do not edit other files, do not run git.

The character: a 1920s gothic "Infantry Ranged Trooper": fedora, glowing orange mask, long dark-blue
double-breasted coat with gold trim, shoulder armour, bracers, gloves, belt, tattered underskirt,
trousers, strapped boots. It is one fused sculpted mesh; your job is to say what separate things a
modeller would build.

## Evidence (view with the Read tool; look at ALL of the images for your region)
- `out/survey_<name>_tex.png`: the sculpt with its painted colour, 8 camera angles (labelled).
- `out/survey_<name>_clay.png`: same cameras, plain grey so shape and edges are clear.
- Red tick labels on front/back/left/right panels give coordinates (figure height = 1.0; y is up,
  x is across with the character's LEFT at larger x, z is depth with the front at larger z).
- `reference/front.png`, `reference/side.png`, `reference/back.png`: the design drawings (authority).
- `reference/character-sheet.png`: design sheet with detail call-outs.
Colour is a strong clue: gold/brass = metal trim or hardware, dark blue = coat cloth, near-black =
leather or under-layers, orange = the mask.

## What counts as an object
Anything a modeller would make as its own piece: a garment panel, a plate, a strap, a buckle, a
stud/rivet/button (list a repeated item ONCE with a count and where the copies are), a trim strip,
an emblem, a cord, a seam flap, an ear. Layers count separately (a strap lying on a plate is two
objects). Paint-only marks (scratches, wear, glow) are not objects, but list a painted feature if it
looks like it should be geometry (for example a slit or an engraved line).

## Output
Write `survey/<region>_<your letter>.json`: a JSON list; one entry per object:
{
 "name": "short_snake_case",
 "parent": "what it sits on or belongs to (or null)",
 "count": 1,
 "side": "centre | left | right | both",
 "material": "cloth | leather | metal | other",
 "rigid": true,
 "where": "plain description plus approximate coordinates/extent read from the tick labels",
 "shape": "one sentence: what it looks like",
 "seen_in": ["image file + panel names where it is clearest"],
 "in_reference": "yes | no | differs: how",
 "confidence": "high | medium | low",
 "notes": "layering (what is over/under it), anything ambiguous, left/right differences"
}
Be exhaustive and concrete: count studs and buttons carefully per side, say when left and right
differ, and include things you are unsure about with confidence "low" rather than leaving them out.
Also list anything the references show that the sculpt lacks.
Final message: the path of the file you wrote, the number of entries, and your 3 least certain items.
