from pathlib import Path
import cv2
import numpy as np

FEATURE_NAMES = ["area_px","perimeter_px","major_axis_px","minor_axis_px","aspect_ratio",
                 "solidity","extent","skeleton_length_px","branch_points","mean_width_px","width_std_px"]

def skeletonize(mask):
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while cv2.countNonZero(img):
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
    return skel

def extract_geometry(mask, scale_mm_per_pixel=None):
    if isinstance(mask, (str, Path)):
        mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if mask is None: raise ValueError("Mask could not be read")
    binary = (mask > 0).astype(np.uint8)
    area = float(binary.sum())
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
    major = minor = aspect = solidity = extent = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) >= 5:
            (_, _), (w,h), _ = cv2.fitEllipse(cnt)
            major, minor = float(max(w,h)), float(max(1e-6,min(w,h)))
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            major, minor = float(max(w,h)), float(max(1,min(w,h)))
        aspect = major / max(minor, 1e-6)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = float(cv2.contourArea(cnt) / max(hull_area, 1e-6))
        x,y,w,h = cv2.boundingRect(cnt)
        extent = float(cv2.contourArea(cnt) / max(w*h, 1))
    skel = skeletonize(binary)
    skel_len = float((skel > 0).sum())
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    widths = dist[skel > 0] * 2.0
    mean_width = float(widths.mean()) if widths.size else 0.0
    std_width = float(widths.std()) if widths.size else 0.0
    kernel = np.ones((3,3), np.uint8); kernel[1,1] = 0
    neighbors = cv2.filter2D((skel > 0).astype(np.uint8), -1, kernel)
    branches = float(((skel > 0) & (neighbors >= 3)).sum())
    vals = [area,perimeter,major,minor,aspect,solidity,extent,skel_len,branches,mean_width,std_width]
    out = dict(zip(FEATURE_NAMES, map(float, vals)))
    out["scale_mm_per_pixel"] = float(scale_mm_per_pixel) if scale_mm_per_pixel else None
    out["area_mm2"] = area*scale_mm_per_pixel**2 if scale_mm_per_pixel else None
    out["length_mm"] = skel_len*scale_mm_per_pixel if scale_mm_per_pixel else None
    out["mean_width_mm"] = mean_width*scale_mm_per_pixel if scale_mm_per_pixel else None
    return out

def geometry_vector(mask_path, scale_mm_per_pixel=None):
    d = extract_geometry(mask_path, scale_mm_per_pixel)
    return np.array([d[k] for k in FEATURE_NAMES], dtype=np.float32)
