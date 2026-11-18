import argparse, json, random, shutil
from pathlib import Path
from PIL import Image
import yaml

p=argparse.ArgumentParser()
p.add_argument("--json",required=True); p.add_argument("--images",required=True)
p.add_argument("--out",required=True); p.add_argument("--task",choices=["detect","segment"],default="detect")
p.add_argument("--seed",type=int,default=42); a=p.parse_args()
d=json.load(open(a.json)); cats=sorted(d["categories"],key=lambda x:x["id"])
catmap={c["id"]:i for i,c in enumerate(cats)}; images={x["id"]:x for x in d["images"]}
anns={}
for x in d["annotations"]: anns.setdefault(x["image_id"],[]).append(x)
ids=list(images); random.Random(a.seed).shuffle(ids); n=len(ids)
splits={"train":ids[:int(.7*n)],"val":ids[int(.7*n):int(.85*n)],"test":ids[int(.85*n):]}
out=Path(a.out); src=Path(a.images)
for split,items in splits.items():
 for iid in items:
  im=images[iid]; sp=src/im["file_name"]
  if not sp.exists(): raise FileNotFoundError(sp)
  dst=out/"images"/split/sp.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(sp,dst)
  W,H=Image.open(sp).size; lines=[]
  for an in anns.get(iid,[]):
   x,y,w,h=an["bbox"]; cls=catmap[an["category_id"]]
   if a.task=="detect": lines.append(f"{cls} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
   else:
    seg=an.get("segmentation",[])
    if not isinstance(seg,list) or not seg or not isinstance(seg[0],list): continue
    for poly in seg:
     if len(poly)<6: continue
     coords=[]
     for j in range(0,len(poly),2): coords += [poly[j]/W,poly[j+1]/H]
     lines.append(str(cls)+" "+" ".join(f"{v:.6f}" for v in coords))
  lab=out/"labels"/split/(sp.stem+".txt"); lab.parent.mkdir(parents=True,exist_ok=True); lab.write_text("\n".join(lines))
cfg={"path":str(out.resolve()),"train":"images/train","val":"images/val","test":"images/test",
     "names":{i:c["name"] for i,c in enumerate(cats)}}
(out/"data.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
print(out/"data.yaml")
