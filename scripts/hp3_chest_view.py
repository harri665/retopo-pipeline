"""view only chest-region labels (others merged to unlabelled) so colours stay distinct"""
import sys, numpy as np
d = np.load("survey/labels_chest.npz"); lab = d["lab"]; names = list(d["names"])
keep = ["coat_upper","lapel_l","lapel_r","collar","neck_wrap","belt","sleeve_l","sleeve_r","armhole_roll_l","armhole_roll_r","shoulder_arch_l","shoulder_arch_r","shoulder_cap_l","shoulder_cap_r","shoulder_flap_l","shoulder_flap_r","shoulder_front_strap_l","shoulder_front_strap_r"]
keep=[k for k in keep if k in names]
new = -np.ones_like(lab)
for i,k in enumerate(keep): new[lab==names.index(k)] = i
np.savez_compressed("survey/labels_chest_view.npz", lab=new.astype(np.int16), names=np.array(keep))
