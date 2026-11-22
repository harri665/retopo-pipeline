from scripts.hp_d_common import *
T = 3.5 * mm
if __name__ == "__main__":
    V, Q = ring_object(CUFF_X0, CUFF_X1, r_cuff, lambda x: r_cuff(x) - T, 0.8 * mm, 5, N=56)
    emit("bracer_gold_cuff", V, Q)
