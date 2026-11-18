import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("--weights",required=True); p.add_argument("--data",required=True)
p.add_argument("--split",default="test"); a=p.parse_args()
m=YOLO(a.weights); r=m.val(data=a.data,split=a.split); print(r.results_dict)
