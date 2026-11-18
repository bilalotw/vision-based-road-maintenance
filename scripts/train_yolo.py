import argparse, sys, yaml
from pathlib import Path
from ultralytics import YOLO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from road_distress.reproducibility import set_seed, save_environment
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/yolo.yaml"); a=p.parse_args()
cfg=yaml.safe_load(open(a.config)); set_seed(cfg["seed"])
model=YOLO(cfg["model"],task=cfg["task"])
model.train(data=cfg["data"],epochs=cfg["epochs"],imgsz=cfg["imgsz"],batch=cfg["batch"],
 workers=cfg["workers"],device=cfg["device"],patience=cfg["patience"],optimizer=cfg["optimizer"],
 lr0=cfg["lr0"],weight_decay=cfg["weight_decay"],amp=cfg["amp"],cache=cfg["cache"],
 project=cfg["project"],name=cfg["name"],seed=cfg["seed"],deterministic=True)
run=Path(cfg["project"])/cfg["name"]; save_environment(run/"environment.json")
(run/"config_snapshot.yaml").write_text(Path(a.config).read_text())
