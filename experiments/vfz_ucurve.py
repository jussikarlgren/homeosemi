import os,sys,pickle,statistics as st
sys.path.insert(0,"nanoGPT"); sys.path.insert(0,"experiments")
import numpy as np
from eval_blimp import load_model
from verb_nll_eval import per_verb_nll
from verb_regularity import get_regularity
runs="experiments/runs"; SEEDS=[1337,42,2024]
meta=pickle.load(open("experiments/data/babylm_lit/meta.pkl","rb"))
stoi=meta["stoi"]; reg,_=get_regularity(meta)
vid={stoi[w]:w for w in reg if w in stoi}
val=np.memmap("experiments/data/babylm_lit/val.bin",dtype=np.uint16,mode="r")
def buckets(ckpt):
    m,_=load_model(ckpt,"cpu"); nll,cnt=per_verb_nll(m,val,set(vid),128,"cpu",600000)
    r=[nll[v] for v in nll if cnt.get(v,0)>=30 and reg[vid[v]]["label"]=="regular"]
    ir=[nll[v] for v in nll if cnt.get(v,0)>=30 and reg[vid[v]]["label"]=="irregular"]
    return st.mean(r),st.mean(ir)
def avg(fmt):
    R=[];I=[]
    for s in SEEDS:
        r,i=buckets(fmt.format(s=s)); R.append(r); I.append(i)
    return (st.mean(R),st.pstdev(R)),(st.mean(I),st.pstdev(I))
for enc in ["levin","frame"]:
    for bud in ["1M","100k"]:
        base=f"{runs}/word_{enc}_vfz"
        smid=avg(f"{base}_staged_lit_{bud}_s{{s}}_mid_ckpt")
        sfin=avg(f"{base}_staged_lit_{bud}_s{{s}}_ckpt")
        afin=avg(f"{base}_always_lit_{bud}_s{{s}}_ckpt")
        print(f"\n[{enc} {bud}] reg / irr complement-NLL (mean+/-sd, 3 seeds)",flush=True)
        print(f"  staged MID (frozen)     reg {smid[0][0]:.3f}  irr {smid[1][0]:.3f}")
        print(f"  staged FINAL(released)  reg {sfin[0][0]:.3f}  irr {sfin[1][0]:.3f}")
        print(f"  always FINAL(frozen)    reg {afin[0][0]:.3f}  irr {afin[1][0]:.3f}")
        print(f"  RELEASE effect (staged-always,final):  reg {sfin[0][0]-afin[0][0]:+.3f}  irr {sfin[1][0]-afin[1][0]:+.3f}")
        print(f"  => irr-specific release gain (irr-reg): {(sfin[1][0]-afin[1][0])-(sfin[0][0]-afin[0][0]):+.3f}")
print("\nDONE",flush=True)
