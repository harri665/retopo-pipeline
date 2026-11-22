"""shoulder_arch_strap_l / _r (group C). Shared model in scripts/hp_c_shoulder.py + hp_c_common.py.
Writes hp_cages/shoulder_arch_strap_{l,r}.obj and hipoly/shoulder_arch_strap_{l,r}.obj (cage subdivided 2 levels, creased rims)."""
from scripts.hp_c_shoulder import objects
from scripts.hp_c_common import save_pair
if __name__ == "__main__":
    n = "shoulder_arch_strap"; V, Q = objects()[n]; print(n, save_pair(n, V, Q))
