"""boot_toe_cap (hybrid): fitted domed cap = the vamp cage from the seam (u=0.074) to the toe, offset outward 2.4 mm and solidified.
Needs hp_cages/boot_upper_l.obj."""
import numpy as np
from retopo import hipoly
from scripts.hp_f_common import *

U_SEAM = 0.074
OFF, THICK = 0.0024, 0.0030


def main():
    V, Q = meshio.load_obj("hp_cages/boot_upper_l.obj")
    Lc = toL(V); N = hipoly.vertex_normals(V, Q)
    sel = [q for q in Q if (Lc[q, 0] > U_SEAM).all()]
    used = sorted(set(i for q in sel for i in q)); mp = {o: i for i, o in enumerate(used)}
    Vs = V[used] + N[used] * OFF; Qs = [[mp[i] for i in q] for q in sel]
    Vs, Qs = B.solidify(Vs, fix_winding(Qs), THICK)
    n = save_object("boot_toe_cap_l", Vs, Qs, crease_angle=35, levels=2)
    print("toe cap", n, "cage quads", len(Qs))


if __name__ == "__main__":
    main()
