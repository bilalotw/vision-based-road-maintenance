import numpy as np
from road_distress.geometry import extract_geometry
def test_geometry():
 m=np.zeros((80,100),dtype=np.uint8); m[35:43,10:90]=255
 d=extract_geometry(m)
 assert d["area_px"]>0 and d["perimeter_px"]>0 and d["mean_width_px"]>0
