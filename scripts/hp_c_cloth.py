import numpy as np
from retopo import build as B, meshio
from scripts.arms_pauldron import axis_pt
def mk(n):
    V,P=meshio.load_obj(f"hipoly/{n}.obj"); F,_=meshio.triangulate(P); return B.Caster(V,F)
cl=[mk("sleeve_l"),mk("coat_upper")]
for n in ("shoulder_arch_strap_l","shoulder_cap_plate_l","shoulder_lame_flap_l"):
    V,_=meshio.load_obj(f"hp_cages/{n}.obj"); Vi=V[len(V)//2:]
    gaps=[];pos=[]
    for p in Vi:
        a=axis_pt(p[0]); d=p-a; d[0]=0; r=np.linalg.norm(d); d/=r
        best=None
        for C in cl:
            h=C.first(a+d*0.2,-d,0.2)   # outermost cloth along the ray
            if h: rc=0.2-h[3]; best=rc if best is None else max(best,rc)
        if best is not None: gaps.append(r-best); pos.append(p)
    g=np.array(gaps); pos=np.array(pos)
    print(n,"n",len(g),"gap (plate inner minus cloth outer): min %.4f p5 %.4f med %.4f p95 %.4f max %.4f"%(g.min(),*np.percentile(g,[5,50,95]),g.max()), "poking(<0):",(g<0).sum())
    w=np.argsort(g)[:3]; print("  worst",np.round(pos[w],3).tolist(),np.round(g[w],4).tolist())
print("--- poking points (gap<0) for flap/cap: x range, theta range")
for n in ("shoulder_cap_plate_l","shoulder_lame_flap_l"):
    V,_=meshio.load_obj(f"hp_cages/{n}.obj"); Vi=V[len(V)//2:]; out=[]
    for p in Vi:
        a=axis_pt(p[0]); d=p-a; d[0]=0; r=np.linalg.norm(d); d/=r; best=None
        for C in cl:
            h=C.first(a+d*0.2,-d,0.2)
            if h: rc=0.2-h[3]; best=rc if best is None else max(best,rc)
        if best is not None and r-best<0: out.append([p[0],np.degrees(np.arctan2(d[2],d[1])),r-best])
    o=np.array(out); print(n,"x %.3f-%.3f theta %.0f..%.0f worst %.4f"%(o[:,0].min(),o[:,0].max(),o[:,1].min(),o[:,1].max(),o[:,2].min()))
    v=o[o[:,0]>0.615]; print("visible x>0.615: n",len(v), "worst", (v[:,2].min() if len(v) else None), np.round(v[np.argsort(v[:,2])[:5]],3).tolist())
