import json, math, os, time
import numpy as np
import torch, torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
import pandas as pd

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

d = np.load("/Users/rls/ecs271/data/sc/combat/combat_paired.npz", allow_pickle=True)
bulk_raw = d["bulk_x"].astype(np.float32); sc_raw = np.log1p(d["sc_x"].astype(np.float32))
donors = d["donors"]; N = len(donors)
def zscore(x): return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-8)
top_idx = np.argsort(bulk_raw.var(0))[-2000:]
bh = zscore(bulk_raw[:,top_idx]); sh = zscore(sc_raw[:,top_idx]); G = bh.shape[1]
clin = pd.read_csv("/Users/rls/ecs271/data/sc/combat/CBD-KEY-CLINVAR/COMBAT_CLINVAR_for_processed.txt",sep="\t")
sev_map = clin.drop_duplicates("COMBAT_ID").set_index("COMBAT_ID")
sev = sev_map["Hospitalstay"].reindex(donors).fillna(sev_map["Hospitalstay"].median()).values.astype(np.float32)
sev = (sev-sev.min())/(sev.max()-sev.min()+1e-8)

class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(G,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,512),nn.LayerNorm(512),nn.GELU(),nn.Linear(512,256),nn.LayerNorm(256),nn.GELU(),nn.Linear(256,128))
    def forward(self,x): return self.net(x)

def encode(enc,x): 
    enc.eval()
    with torch.no_grad(): return enc(torch.tensor(x,dtype=torch.float32).to(DEVICE)).cpu().numpy()

def train_fold(bulk_np, sc_np, sev_np, gamma, epochs=300):
    enc = Enc().to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bt = torch.tensor(bulk_np,dtype=torch.float32); st = torch.tensor(sc_np,dtype=torch.float32)
    s = sev_np[:,None]; sev_w = torch.tensor(1+gamma*(1-np.abs(s-s.T)),dtype=torch.float32).to(DEVICE)
    mask = torch.eye(len(bt),dtype=torch.bool,device=DEVICE)
    for ep in range(epochs):
        tau = 0.05 + 0.25*(1-ep/epochs)
        enc.train()
        zb = enc(bt.to(DEVICE)*(torch.rand(bt.shape,device=DEVICE)>0.3).float())
        zs = enc(st.to(DEVICE))
        z1n=F.normalize(zb,dim=1); z2n=F.normalize(zs,dim=1)
        s12=(z1n@z2n.T)/tau; s11=(z1n@z1n.T)/tau; s22=(z2n@z2n.T)/tau
        s12=s12+torch.where(~mask,torch.log(sev_w.clamp(1e-8)),torch.zeros_like(sev_w))
        pos=s12.diagonal()
        l1=(-pos+torch.logsumexp(torch.cat([s12.masked_fill(mask,-1e9),s11.masked_fill(mask,-1e9)],1),1)).mean()
        l2=(-pos+torch.logsumexp(torch.cat([s12.T.masked_fill(mask,-1e9),s22.masked_fill(mask,-1e9)],1),1)).mean()
        opt.zero_grad(); ((l1+l2)/2).backward()
        nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step(); sch.step()
    return enc

rng=np.random.default_rng(42); idx=rng.permutation(N); folds=np.array_split(idx,5)

print(f"{'γ':>6}  {'argmax':>8}  {'hungarian':>10}")
results={}
for gamma in [3.0, 5.0, 8.0, 15.0]:
    arg_f=[]; hun_f=[]
    for fi,test_idx in enumerate(folds):
        tr=np.concatenate([folds[j] for j in range(5) if j!=fi])
        enc=train_fold(bh[tr],sh[tr],sev[tr],gamma)
        zb=encode(enc,bh); zs=encode(enc,sh[test_idx])
        zb_=F.normalize(torch.tensor(zb,dtype=torch.float32),dim=1)
        zs_=F.normalize(torch.tensor(zs,dtype=torch.float32),dim=1)
        sim=(zs_@zb_.T).numpy()
        nn_idx=sim.argmax(1)
        arg_f.append(sum(nn_idx[i]==test_idx[i] for i in range(len(test_idx)))/len(test_idx))
        r,c=linear_sum_assignment(-sim[:,test_idx])
        hun_f.append(sum(test_idx[c[i]]==test_idx[i] for i in range(len(test_idx)))/len(test_idx))
    ma=float(np.mean(arg_f)); mh=float(np.mean(hun_f))
    print(f"{gamma:>6.1f}  {ma:>8.3f}  {mh:>10.3f}  folds_hun={[round(x,3) for x in hun_f]}")
    results[gamma]={"argmax":ma,"hungarian":mh}

os.makedirs("/Users/rls/ecs271/vae_health/analysis/results",exist_ok=True)
with open("/Users/rls/ecs271/vae_health/analysis/results/rq1_gamma_sweep.json","w") as f:
    json.dump({"results":{str(k):v for k,v in results.items()}},f,indent=2)
print("Saved → rq1_gamma_sweep.json")
