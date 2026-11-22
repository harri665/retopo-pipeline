import numpy as np
from retopo import meshio, build as B
Vc,Pc=meshio.load_obj("parts/collar.obj"); Fc,_=meshio.triangulate(Pc)
C=B.Caster(Vc,Fc.astype(int))
V,P=meshio.load_obj("parts/coat_upper.obj")
top=np.where(np.isclose(V[:,1],0.8475,atol=1e-4))[0]
ax=np.array([0.4645,0,0.14]); print("collar y range",Vc[:,1].min(),Vc[:,1].max(),"top-ring verts",len(top))
worst=9
for i in top:
    d=V[i]-ax; d[1]=0; r=np.linalg.norm(d); d/=r
    o=np.array([ax[0],V[i,1],ax[2]])
    hs=C.all(o,d,0.2)
    rin=hs[0][0] if hs else None
    m=(rin-r) if rin else None
    if m is not None: worst=min(worst,m)
    if m is None or m<0.003: print("vertex",np.round(V[i],3),"r %.4f collar inner %s"%(r,None if rin is None else round(rin,4)))
print("min margin inside collar",worst)
# also rows below: any coat vertex (y>0.80) outside the collar's inner layer at the same y?
print("-- all vertices y>0.79: coat radius vs collar hits along ray (first=inner, last=outer)")
bad=0
for i in np.where(V[:,1]>0.79)[0]:
    d=V[i]-ax; d[1]=0; r=np.linalg.norm(d)
    if r<1e-6: continue
    d/=r; o=np.array([ax[0],V[i,1],ax[2]]); hs=C.all(o,d,0.2)
    if hs and r>hs[0][0]-0.002 and abs(V[i,0]-0.4645)<0.1:
        bad+=1; print(np.round(V[i],3),"r %.4f inner %.4f outer %.4f"%(r,hs[0][0],hs[-1][0]))
print("coat vertices not inside collar inner by 2mm:",bad)
