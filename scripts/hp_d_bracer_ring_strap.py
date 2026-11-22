from scripts.hp_d_common import *
if __name__ == "__main__":
    V, Q = ring_object(STRAP_X0, STRAP_X1, lambda x: R_STRAP + 0 * x, lambda x: R_STRAP - 2.8 * mm, 0.6 * mm, 2, N=48)
    emit("bracer_ring_strap", V, Q)
