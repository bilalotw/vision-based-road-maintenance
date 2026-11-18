import argparse, sys
from pathlib import Path
import torch, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from road_distress.datasets import SeverityManifestDataset
from road_distress.models.severity_net import SeverityNet, mc_predict
from road_distress.evaluation import severity_metrics, expected_calibration_error
p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--manifest",required=True)
p.add_argument("--batch-size",type=int,default=16); p.add_argument("--passes",type=int,default=30)
p.add_argument("--output",default="outputs/severity_eval"); a=p.parse_args()
ck=torch.load(a.checkpoint,map_location="cpu"); ds=SeverityManifestDataset(a.manifest)
loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False)
net=SeverityNet(len(ck["classes"]),use_image=ck["use_image"],use_geometry=ck["use_geometry"],pretrained=False)
net.load_state_dict(ck["model"]); dev="cuda" if torch.cuda.is_available() else "cpu"; net.to(dev)
ys=[]; preds=[]; probs=[]; unc=[]
for b in loader:
 r=mc_predict(net,b["image"].to(dev),b["geometry"].to(dev),a.passes)
 probs.extend(r["probabilities"].cpu().numpy()); preds.extend(r["probabilities"].argmax(1).cpu().numpy())
 ys.extend(b["label"].numpy()); unc.extend(r["predictive_entropy"].cpu().numpy())
metrics=severity_metrics(ys,preds); metrics["ece"]=expected_calibration_error(probs,ys)
metrics["mean_predictive_entropy"]=float(np.mean(unc))
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
(out/"metrics.json").write_text(__import__("json").dumps(metrics,indent=2))
ConfusionMatrixDisplay(confusion_matrix(ys,preds),display_labels=ck["classes"]).plot(cmap="Blues",xticks_rotation=30)
plt.tight_layout(); plt.savefig(out/"confusion_matrix.png",dpi=200); plt.close()
print(metrics)
