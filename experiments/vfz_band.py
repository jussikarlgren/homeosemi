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
print("irregular-specific release gain at 1M, per seed and mean+/-sd")
print("(negative = releasing the freeze helps irregular verbs MORE than regular = U-curve)")
for enc in ["levin","frame"]:
    gains=[]
    for s in SEEDS:
        sr,si=buckets(f"{runs}/word_{enc}_vfz_staged_lit_1M_s{s}_ckpt")
        ar,ai=buckets(f"{runs}/word_{enc}_vfz_always_lit_1M_s{s}_ckpt")
        gains.append((si-ai)-(sr-ar))
    print(f"  {enc:6s} 1M:  per-seed {['%+.3f'%g for g in gains]}  mean {st.mean(gains):+.3f} +/- {st.pstdev(gains):.3f}")
print("DONE")
