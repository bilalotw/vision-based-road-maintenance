import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from road_distress.datasets import rdd_csv_to_yolo
p=argparse.ArgumentParser()
p.add_argument("--images",required=True); p.add_argument("--csv",required=True)
p.add_argument("--out",required=True); p.add_argument("--class-map",required=True)
p.add_argument("--seed",type=int,default=42)
a=p.parse_args()
print(rdd_csv_to_yolo(a.csv,a.images,a.out,a.class_map,a.seed))
