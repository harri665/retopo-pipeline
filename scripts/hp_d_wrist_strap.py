from scripts.hp_d_common import *
if __name__ == "__main__":
    V, Q = ring_object(WS_X0, WS_X1, r_ws, lambda x: r_tube(x) - 0.3 * mm, 0.6 * mm, 3, N=40)
    emit("wrist_strap", V, Q)
