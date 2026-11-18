import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("--weights",required=True); p.add_argument("--image",required=True)
p.add_argument("--task",default="detect",choices=["detect","segment"]); p.add_argument("--conf",type=float,default=.25)
p.add_argument("--output",default="outputs/predictions"); a=p.parse_args()
model=YOLO(a.weights,task=a.task)
model.predict(source=a.image,conf=a.conf,save=True,project=a.output,name="run")
