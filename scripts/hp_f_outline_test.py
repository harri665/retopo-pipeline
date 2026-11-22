import numpy as np
from PIL import Image, ImageDraw
from scripts.hp_f_common import *
for y in (0.003,0.008,0.012):
    p=trace_outline(y); print(y,len(p),p[:,0].min(),p[:,0].max(),p[:,1].min(),p[:,1].max())
    np.save('/tmp/claude-0/-home-user-retopo-pipeline/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/f/ol_%g.npy'%y,p)
im=Image.new('RGB',(900,600),'white'); d=ImageDraw.Draw(im)
for y,col in ((0.003,'red'),(0.008,'blue'),(0.012,'green')):
    p=np.load('/tmp/claude-0/-home-user-retopo-pipeline/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/f/ol_%g.npy'%y)
    q=[(60+(a+0.045)*5000, 300-(b)*5000) for a,b in p]
    d.line(q+[q[0]],fill=col,width=2)
    for x in q: d.ellipse([x[0]-2,x[1]-2,x[0]+2,x[1]+2],fill=col)
im.save('/tmp/claude-0/-home-user-retopo-pipeline/c1d6d7cb-a765-5745-a8ad-9f8025350a01/scratchpad/f/ol.png')
