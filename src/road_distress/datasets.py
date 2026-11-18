from pathlib import Path
import shutil
import cv2, yaml, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

def read_class_map(path):
    with open(path) as f: return yaml.safe_load(f)

def rdd_csv_to_yolo(csv_path, images_root, output_root, class_map, seed=42, train=0.70, val=0.15):
    df = pd.read_csv(csv_path)
    required = {"image","class","xmin","ymin","xmax","ymax"}
    if not required.issubset(df.columns): raise ValueError(f"CSV needs columns {required}")
    cfg = read_class_map(class_map); names = cfg["canonical_classes"]
    mapping = cfg["source_to_canonical"]; ids = {n:i for i,n in enumerate(names)}
    unique = sorted(df.image.astype(str).unique())
    rng = np.random.default_rng(seed); rng.shuffle(unique)
    n=len(unique); splits={"train":unique[:int(n*train)],"val":unique[int(n*train):int(n*(train+val))],
                           "test":unique[int(n*(train+val)):]}
    imgroot=Path(images_root); out=Path(output_root)
    for split, images in splits.items():
        for name in images:
            src=imgroot/name
            if not src.exists(): raise FileNotFoundError(src)
            dst=out/"images"/split/src.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
            im=cv2.imread(str(src)); h,w=im.shape[:2]; lines=[]
            for r in df[df.image.astype(str)==name].itertuples(index=False):
                cname=mapping.get(str(r[1]))
                if cname not in ids: continue
                x1,y1,x2,y2=map(float,(r.xmin,r.ymin,r.xmax,r.ymax))
                x1=max(0,min(w,x1)); x2=max(0,min(w,x2)); y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
                if x2<=x1 or y2<=y1: continue
                lines.append(f"{ids[cname]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            lab=out/"labels"/split/(src.stem+".txt"); lab.parent.mkdir(parents=True,exist_ok=True)
            lab.write_text("\n".join(lines))
    data={"path":str(out.resolve()),"train":"images/train","val":"images/val","test":"images/test",
          "names":{i:n for i,n in enumerate(names)}}
    (out/"data.yaml").write_text(yaml.safe_dump(data,sort_keys=False))
    return out/"data.yaml"

class SeverityManifestDataset(Dataset):
    def __init__(self, manifest, image_size=224, geometry_mode=True):
        self.df=pd.read_csv(manifest).fillna(""); self.base=Path(manifest).resolve().parent
        if not {"image","severity"}.issubset(self.df.columns): raise ValueError("Manifest needs image,severity")
        self.image_size=image_size; self.geometry_mode=geometry_mode
        self.classes=sorted(self.df.severity.astype(int).unique().tolist())
        self.class_to_idx={c:i for i,c in enumerate(self.classes)}
        if len(self.classes)<2: raise ValueError("Need at least two severity classes")
    def resolve(self, value):
        p=Path(str(value)); return p if p.is_absolute() else self.base/p
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        from .geometry import extract_geometry, FEATURE_NAMES
        r=self.df.iloc[i]; ip=self.resolve(r.image)
        img=cv2.imread(str(ip))
        if img is None: raise FileNotFoundError(ip)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); img=cv2.resize(img,(self.image_size,self.image_size))
        img=torch.from_numpy(img.transpose(2,0,1).copy()).float()/255.
        geom=np.zeros(len(FEATURE_NAMES),np.float32)
        mask_value=str(r.get("mask","")).strip()
        if self.geometry_mode and mask_value:
            mp=self.resolve(mask_value); mask=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
            if mask is None: raise FileNotFoundError(mp)
            mask=cv2.resize(mask,(self.image_size,self.image_size),interpolation=cv2.INTER_NEAREST)
            scale=str(r.get("scale_mm_per_pixel","")).strip()
            scale=float(scale) if scale else None
            d=extract_geometry(mask,scale); geom=np.array([d[k] for k in FEATURE_NAMES],np.float32)
        geom=np.sign(geom)*np.log1p(np.abs(geom))
        return {"image":img,"geometry":torch.tensor(geom),"label":torch.tensor(self.class_to_idx[int(r.severity)]),
                "road_id":str(r.get("road_id",i)),"image_path":str(ip)}
