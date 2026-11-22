import numpy as np, json
from scipy.interpolate import RegularGridInterpolator as RGI
from retopo import meshio
from scripts.arms_pauldron import axis_pt
xs=0.53+0.0015*np.arange(107); th=-115+0.75*np.arange(307)
R=np.load("cache/hp_c_R.npy"); R=np.nan_to_num(R,nan=0.0)
f=RGI((xs,th),R,bounds_error=False,fill_value=np.nan)
bumps=[b for b in json.load(open("survey/bumps.json"))]
for n in ("shoulder_arch_strap_l","shoulder_cap_plate_l","shoulder_lame_flap_l","shoulder_edge_trim_l"):
    V,P=meshio.load_obj(f"hipoly/{n}.obj"); V=V[np.unique(np.concatenate(P))]
    ax=np.array([axis_pt(x) for x in V[:,0]]); d=V-ax; r=np.hypot(d[:,1],d[:,2]); t=np.degrees(np.arctan2(d[:,2],d[:,1]))
    s=f(np.stack([V[:,0],t],1)); dev=r-s
    ok=np.isfinite(dev)&(s>0)
    for b in bumps:
        ok&=np.linalg.norm(V-np.array(b["centre"]),axis=1)>b["radius"]*1.8+0.003
    # outer surface = the larger-r of the two layers: take upper half of dev
    dv=dev[ok]; hi=dv[dv>np.percentile(dv,50)] if False else dv
    print(n,"dev r-sculpt: p5 %.4f p50 %.4f p95 %.4f max %.4f"%tuple(np.percentile(dv,[5,50,95,100])), "n",ok.sum())
print("--- outer cage verts")
for n in ("shoulder_arch_strap_l","shoulder_cap_plate_l","shoulder_lame_flap_l"):
    V,P=meshio.load_obj(f"hp_cages/{n}.obj"); V=V[:len(V)//2]
    ax=np.array([axis_pt(x) for x in V[:,0]]); d=V-ax; r=np.hypot(d[:,1],d[:,2]); t=np.degrees(np.arctan2(d[:,2],d[:,1]))
    s=f(np.stack([V[:,0],t],1)); dev=r-s; ok=np.isfinite(dev)&(s>0)
    for b in bumps: ok&=np.linalg.norm(V-np.array(b["centre"]),axis=1)>b["radius"]*1.8+0.003
    big=np.where(ok&(np.abs(dev)>0.004))[0]
    print(n,"p5 %.4f p50 %.4f p95 %.4f"%tuple(np.percentile(dev[ok],[5,50,95])),"n>4mm",len(big),"of",ok.sum())
    if len(big): print("  at theta,x:",np.round(np.c_[t[big],V[big,0],dev[big]][:12],4).tolist())
