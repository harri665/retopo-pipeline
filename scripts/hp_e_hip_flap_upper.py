from scripts.hp_e_hips import *
if __name__ == "__main__":
    Pu, (V, Q) = upper_flap()
    V, Q = mirrored(V, Q)
    print("hip_flap_upper", save_cage_and_hipoly("hip_flap_upper", V, Q, crease_angle=45, crease=0.85))
