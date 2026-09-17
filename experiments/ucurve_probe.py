import os,sys,pickle,statistics as st
sys.path.insert(0,"nanoGPT"); sys.path.insert(0,"experiments")
import numpy as np
from eval_blimp import load_model
from verb_nll_eval import per_verb_nll
from verb_regularity import get_regularity
runs="experiments/runs"; SEEDS=[1337,42,2024]
meta=pickle.load(open("experiments/data/babylm_lit/meta.pkl","rb"))
stoi=meta["stoi"]; reg,_=get_regularity(meta)
verb_ids={stoi[w]:w for w in reg if w in stoi}
val=np.memmap("experiments/data/babylm_lit/val.bin",dtype=np.uint16,mode="r")
def bucket_nll(ckpt):
    m,_=load_model(ckpt,"cpu")
    nll,cnt=per_verb_nll(m,val,set(verb_ids),128,"cpu",600000)
    r=[nll[v] for v in nll if cnt.get(v,0)>=30 and reg[verb_ids[v]]["label"]=="regular"]
    ir=[nll[v] for v in nll if cnt.get(v,0)>=30 and reg[verb_ids[v]]["label"]=="irregular"]
    return st.mean(r),st.mean(ir)
def avg(fmt):
    rs=[];irs=[]
    for s in SEEDS:
        r,ir=bucket_nll(fmt.format(s=s)); rs.append(r); irs.append(ir)
    return st.mean(rs),st.mean(irs)
for bud in ["100k","1M"]:
    print(f"\n[{bud}] complement-NLL (reg / irr), 3-seed mean:",flush=True)
    sm=avg(f"{runs}/word_frame_staged50_lit_{bud}_s{{s}}_mid_ckpt")
    sf=avg(f"{runs}/word_frame_staged50_lit_{bud}_s{{s}}_ckpt")
    af=avg(f"{runs}/word_frame_lit_{bud}_s{{s}}_ckpt")
    print(f"  staged MID (frozen, it500)    reg {sm[0]:.3f}  irr {sm[1]:.3f}  (irr-reg {sm[1]-sm[0]:+.3f})")
    print(f"  staged FINAL(released,it1000) reg {sf[0]:.3f}  irr {sf[1]:.3f}  (irr-reg {sf[1]-sf[0]:+.3f})")
    print(f"  always FINAL(frozen, it1000)  reg {af[0]:.3f}  irr {af[1]:.3f}  (irr-reg {af[1]-af[0]:+.3f})")
    print(f"  release effect (staged-always,final): reg {sf[0]-af[0]:+.3f}  irr {sf[1]-af[1]:+.3f}")
    print(f"  mid->final trajectory (staged):       reg {sf[0]-sm[0]:+.3f}  irr {sf[1]-sm[1]:+.3f}")
print("\nDONE",flush=True)
