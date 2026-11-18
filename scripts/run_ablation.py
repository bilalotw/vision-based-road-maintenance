import argparse, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output",default="outputs/ablations")
p.add_argument("--epochs",type=int,default=30); a=p.parse_args()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
for name,flag in [("image_only","--image-only"),("geometry_only","--geometry-only"),("image_geometry",None)]:
 cmd=[sys.executable,"scripts/train_severity.py","--manifest",a.manifest,"--output",str(out/name),"--epochs",str(a.epochs)]
 if flag: cmd.append(flag)
 subprocess.run(cmd,check=True)
