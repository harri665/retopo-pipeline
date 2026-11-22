"""Build every labelled object from survey/labels.npz into hipoly3/ (4 at a time).
python -m scripts.hp3_build_all [name ...]   -> out/hp3/<name>.json, docs/hp3_report.json"""
import sys, os, json, subprocess, re
from concurrent.futures import ThreadPoolExecutor
from retopo import seeds, meshio
RULES = [  # first match wins: (regex, edge_mm, sharp)
    (r"glove|hip_cord|hat_band", 2.0, False), (r"mask", 2.5, True),
    (r"bracer|shoulder_|hip_band|hip_shield|wrist_strap", 2.5, True), (r"belt|boot_|pocket_flap", 2.5, False),
    (r"coat_upper|hat_crown|hat_brim|underskirt", 3.5, False), (r"skirt_", 4.0, False), (r"trousers", 5.0, False), (r".", 3.0, False)]


NAMES = []


def run(n):
    e, sharp = next((e, s) for r, e, s in RULES if re.search(r, n))
    for edge in (e, e * 0.8, e * 1.25):   # QuadriFlow sometimes fails at one density: retry at another
        cmd = [sys.executable, "-m", "retopo.fromseg", n, "--edge", str(edge), "--maxbase", "70000" if n.startswith(("skirt_", "trousers", "underskirt")) else "15000"] + (["--sharp"] if sharp else []) + (["--cover", "studs__" + n] if "studs__" + n in NAMES else [])
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=meshio.ROOT)
        try: return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception: err = (p.stderr or p.stdout)[-300:]
    return {"name": n, "PASS": False, "error": err}


if __name__ == "__main__":
    lab, names = seeds.load(); NAMES[:] = names; todo = [n for n in (sys.argv[1:] or names) if not n.startswith('studs__')]; rep = {}
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(run, todo):
            rep[r["name"]] = r; print(r["name"], "PASS" if r.get("PASS") else "FAIL", r.get("seg_to_obj_p95"), r.get("obj_to_seg_p95"), r.get("quads"), r.get("error", ""), flush=True)
    p = os.path.join(meshio.ROOT, "docs", "hp3_report.json"); old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(rep); json.dump(old, open(p, "w"), indent=1)
    print("objects", len(old), "passing", sum(1 for r in old.values() if r.get("PASS")), "quads", sum(r.get("quads", 0) for r in old.values()))
