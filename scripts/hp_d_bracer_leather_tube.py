from scripts.hp_d_common import *
if __name__ == "__main__":
    V, Q = ring_object(TUBE_X0, TUBE_X1, r_tube, lambda x: r_tube(x) - 3.0 * mm, 0.8 * mm, 10, N=32)
    emit("bracer_leather_tube", V, Q)
