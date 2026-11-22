from scripts.hp_e_hips import *
if __name__ == "__main__":
    Pu, _ = upper_flap(); Ps, _ = shield()
    V, Q = mirrored(*trims(Pu, Ps))
    print("hip_flap_trims", save_cage_and_hipoly("hip_flap_trims", V, Q))
