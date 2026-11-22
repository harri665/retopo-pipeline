import numpy as np
from scripts.hp_c_shoulder import *
P,fits,_=fit_layers(); d=P["cap"]
ns,nv=sizes(d,H); print(ns,nv)
fc=fits["cap"][0]
Po,G=plate_outer_grid(d,fc,H)
print(Po.shape)
# bottom edge row (j=nv): print x, theta, R
for i in range(0,ns+1,1):
    p=Po[i,nv]; print(i,np.round(G[i,nv],3), round(float(np.hypot(*(p[1:]-axis_pt(p[0])[1:]))),4), "row-1:",np.round(G[i,nv-1],3))
V,Q=solid_from_grid(Po); N=Po.shape[0]*Po.shape[1]
Pi=V[N:].reshape(Po.shape); off=Po-Pi
print("thk min/max",np.linalg.norm(off,axis=-1).min(),np.linalg.norm(off,axis=-1).max())
n=grid_normals(Po)
for i in range(12,20): print(i,np.round(n[i,nv],3),np.round(n[i,nv-1],3),np.round(Po[i,nv],4))
