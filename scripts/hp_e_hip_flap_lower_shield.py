from scripts.hp_e_hips import *
if __name__ == "__main__":
    Ps, (V, Q) = shield()
    V, Q = mirrored(V, Q)
    print("hip_flap_lower_shield", save_cage_and_hipoly("hip_flap_lower_shield", V, Q, crease_angle=45, crease=0.85))
