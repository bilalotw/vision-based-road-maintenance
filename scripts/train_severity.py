import argparse, sys, json
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, random_split
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from road_distress.datasets import SeverityManifestDataset
from road_distress.models.severity_net import SeverityNet
from road_distress.reproducibility import set_seed, save_environment
p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output",default="outputs/severity")
p.add_argument("--epochs",type=int,default=40); p.add_argument("--batch-size",type=int,default=16)
p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--seed",type=int,default=42)
p.add_argument("--image-only",action="store_true"); p.add_argument("--geometry-only",action="store_true")
p.add_argument("--workers",type=int,default=4); a=p.parse_args()
if a.image_only and a.geometry_only: raise ValueError("Choose one ablation")
set_seed(a.seed); ds=SeverityManifestDataset(a.manifest)
n=len(ds); ntr=int(.7*n); nv=int(.15*n); nte=n-ntr-nv
tr,va,te=random_split(ds,[ntr,nv,nte],generator=torch.Generator().manual_seed(a.seed))
train=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.workers)
val=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=a.workers)
dev="cuda" if torch.cuda.is_available() else "cpu"
net=SeverityNet(len(ds.classes),use_image=not a.geometry_only,use_geometry=not a.image_only).to(dev)
opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True); best=float("inf")
for ep in range(a.epochs):
 net.train(); tl=[]
 for b in train:
  im=b["image"].to(dev); ge=b["geometry"].to(dev); y=b["label"].to(dev)
  opt.zero_grad(); loss=lossfn(net(im,ge),y); loss.backward(); opt.step(); tl.append(loss.item())
 net.eval(); vl=[]
 with torch.no_grad():
  for b in val: vl.append(lossfn(net(b["image"].to(dev),b["geometry"].to(dev)),b["label"].to(dev)).item())
 score=float(np.mean(vl)) if vl else float("inf")
 print(f"epoch {ep+1}: train={np.mean(tl):.4f} val={score:.4f}")
 if score<best:
  best=score; torch.save({"model":net.state_dict(),"classes":ds.classes,
   "use_image":not a.geometry_only,"use_geometry":not a.image_only,"seed":a.seed},out/"best.pt")
save_environment(out/"environment.json")
(out/"split_warning.txt").write_text("Random split is for baseline smoke tests only. Use road/site grouped splits for publication.")
