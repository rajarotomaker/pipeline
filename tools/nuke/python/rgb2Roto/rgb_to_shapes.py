# Depth-assisted auto-roto for Nuke (Integrated Single Script)
# - Full-res MiDaS Small (GPU preferred: CUDA -> MPS -> CPU fallback)
# - Depth + RGB edge fusion (user weight)
# - Guided refinement if available
# - HARD depth segmentation (kmeans + morphological refinement)
# - Connected components -> merge small -> contour -> resample -> animate roto
# - Motion-based splits, Z-sorted layers
# - Point clamping, temporal smoothing, rigidity
#
# NOTE: Requires: torch, torchvision, opencv-python (and optionally opencv-contrib-python)
# Place in Nuke Script Editor, ensure a Read node is selected, then run function:
#   depth_auto_roto()
#
# If anything errors in your environment we'll iterate.

import nuke
import nuke.rotopaint as rp
import cv2
import numpy as np
import torch
import time
import math
import os
import re
from collections import defaultdict

# ---------------------------
# Configuration / Defaults
# ---------------------------
DEFAULT_MIN_AREA_PX = 2000          # small component threshold (merge-to-nearest)
IOU_THRESH_HIGH = 0.7               # IoU > this => single shape
IOU_THRESH_MED = 0.5                # IoU between med and high => medium split
MAX_POINT_SHIFT = 12.0              # pixels per frame clamp (default)
POINT_SMOOTH_ALPHA = 0.3            # new = old*(1-alpha) + new*alpha
SHAPE_RIGIDITY = 0.6                # 0..1 rigidity clamp
EDGE_WEIGHT_DEFAULT = 0.6           # depth weight in fusion slider (0..1)
EXTRACT_BACKGROUND_BY_DEFAULT = False

# ---------------------------
# Utilities for file sequence resolution
# ---------------------------
def get_seq_filename(node, frame):
    file_knob = node["file"].value()
    if "####" in file_knob:
        file_path = file_knob.replace("####", f"{frame:04d}")
    elif re.search(r"%0\dd", file_knob):
        width = int(re.search(r"%0(\d)d", file_knob).group(1))
        file_path = re.sub(r"%0\dd", f"{frame:0{width}d}", file_knob)
    else:
        file_path = file_knob
    return file_path

def read_frame_numpy(node, frame):
    path = get_seq_filename(node, frame)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img

# ---------------------------
# MiDaS loader & inference (GPU-first)
# ---------------------------
_midas_model = None
_midas_transform = None
_device = None

def detect_device():
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        print("🚀 Using CUDA GPU for MiDaS.")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _device = torch.device("mps")
        print("🍎 Using Apple MPS GPU for MiDaS.")
    else:
        _device = torch.device("cpu")
        print("⚠ No GPU found for MiDaS -> falling back to CPU (slower).")
    return _device

def load_midas():
    global _midas_model, _midas_transform
    if _midas_model is not None:
        return
    dev = detect_device()
    print("🔧 Loading MiDaS Small model (this may take a moment first run)...")
    # First-run will download model weights
    _midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    _midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    _midas_model.to(dev)
    _midas_model.eval()
    print(f"✅ MiDaS loaded on {dev}")

def infer_depth_fullres(bgr_img):
    """
    Input: BGR uint8 HxW x3
    Output: float32 HxW normalized 0..1 depth map (near=1, far=0)
    """
    load_midas()
    dev = detect_device()

    # transform expects RGB numpy
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    inp = _midas_transform(img_rgb).unsqueeze(0).to(dev)
    with torch.no_grad():
        pred = _midas_model(inp)
    # pred often smaller; upsample to image size
    pred = torch.nn.functional.interpolate(pred.unsqueeze(0), size=(bgr_img.shape[0], bgr_img.shape[1]), mode="bicubic", align_corners=False)[0,0].cpu().numpy()
    # normalize
    pred = pred - pred.min()
    pred = pred / (pred.max() + 1e-8)
    return pred.astype(np.float32)

# ---------------------------
# Edge and depth refinement
# ---------------------------
def get_depth_edges(depth_map):
    # depth_map float32 0..1
    # use Scharr for sharper edges
    d = (depth_map * 255.0).astype(np.uint8)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = mag / (mag.max() + 1e-8)
    return mag

def get_rgb_edges(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    # use Canny with automatic thresholds via median
    med = np.median(gray)
    lower = int(max(0, 0.66 * med))
    upper = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lower, upper)
    edges = edges.astype(np.float32) / 255.0
    # smooth a tiny bit
    edges = cv2.GaussianBlur(edges, (3,3), 0)
    return edges

def guided_refine(depth_map, rgb_img):
    """Try to run a guided filter to align depth to RGB edges if available (cv2.ximgproc).
       Fallback: bilateral filter."""
    try:
        import cv2.ximgproc as xip
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        depth_f = depth_map.astype(np.float32)
        # guidedFilter arguments: src, guide, radius, eps
        radius = 5
        eps = 1e-4
        refined = xip.guidedFilter(guide=gray, src=depth_f, radius=radius, eps=eps)
        refined = np.clip(refined, 0.0, 1.0)
        return refined
    except Exception as e:
        # fallback: bilateral filter on depth (preserves edges)
        try:
            d8 = (depth_map*255).astype(np.uint8)
            d_blur = cv2.bilateralFilter(d8, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)/255.0
            return np.clip(d_blur, 0.0, 1.0)
        except Exception as e2:
            print("⚠ guided_refine fallback failed - returning input depth. Errors:", e, e2)
            return depth_map

def fuse_edges(depth_edges, rgb_edges, depth_weight):
    # depth_edges, rgb_edges both 0..1 floats
    w = float(depth_weight)
    fused = (depth_edges * w) + (rgb_edges * (1.0 - w))
    fused = fused / (fused.max() + 1e-8)
    return fused

# ---------------------------
# Segmentation (kmeans via cv2) - HARD mode
# ---------------------------
def auto_k_for_depth(depth_map, k_min=3, k_max=8):
    # for HARD sensitivity, choose a higher k by default
    # simple heuristic: use 5 for most frames, but can be adaptive by histogram peaks
    return min(max(5, k_min), k_max)

def segment_depth_kmeans(depth_map, hard=True):
    """
    Returns masks list sorted near->far (closest first).
    Uses cv2.kmeans for clustering.
    """
    h,w = depth_map.shape
    flat = depth_map.reshape(-1,1).astype(np.float32)
    k = auto_k_for_depth(depth_map, k_min=3, k_max=8)
    # prepare criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    attempts = 3
    flags = cv2.KMEANS_PP_CENTERS
    try:
        _, labels, centers = cv2.kmeans(flat, k, None, criteria, attempts, flags)
        labels = labels.reshape(h,w)
        centers = centers.ravel()
    except Exception as e:
        # fallback: simple thresholds into 4 buckets
        labels = np.floor(depth_map * 4).astype(np.int32)
        centers = np.linspace(depth_map.max(), depth_map.min(), 4)
    # sort clusters by center descending (near->far)
    order = np.argsort(-centers)
    masks = []
    for idx in order:
        m = (labels == idx).astype(np.uint8) * 255
        # morphological cleanup
        m = cv2.medianBlur(m, 5)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        masks.append(m)
    return masks

# ---------------------------
# Connected components & merging small ones
# ---------------------------
def find_components(mask):
    # returns list of component masks uint8
    if mask is None or mask.sum() == 0:
        return []
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    comps = []
    for lab in range(1, num_labels):
        comp = (labels == lab).astype(np.uint8) * 255
        comps.append(comp)
    return comps

def area_px(mask):
    return int((mask>0).sum())

def centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (xs.mean(), ys.mean())

def merge_small_to_nearest(components, min_area_px=DEFAULT_MIN_AREA_PX):
    if not components:
        return []
    areas = [area_px(c) for c in components]
    large_idx = [i for i,a in enumerate(areas) if a >= min_area_px]
    small_idx = [i for i,a in enumerate(areas) if a < min_area_px]

    if not large_idx:
        # nothing large, return union as single component
        union = np.zeros_like(components[0])
        for c in components:
            union = cv2.bitwise_or(union, c)
        return [union]

    merged = [components[i].copy() for i in large_idx]
    large_centroids = [centroid(components[i]) for i in large_idx]

    for si in small_idx:
        sc = centroid(components[si])
        if sc is None:
            continue
        # nearest large
        dists = [math.hypot(sc[0]-lc[0], sc[1]-lc[1]) for lc in large_centroids]
        tgt = int(np.argmin(dists))
        merged[tgt] = cv2.bitwise_or(merged[tgt], components[si])
    return merged

# ---------------------------
# Contour -> resample (reuse functions)
# ---------------------------
def resample(contour, target):
    pts = contour[:, 0, :].astype(np.float32)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    segs = np.sqrt(np.sum((pts[1:] - pts[:-1])**2, axis=1))
    L = np.sum(segs)
    if L < 1e-6:
        return np.repeat(pts[:1], target, axis=0)
    dists = np.linspace(0, L, target, endpoint=False)
    result, acc, i = [], 0.0, 0
    for d in dists:
        while i < len(segs)-1 and acc + segs[i] < d:
            acc += segs[i]
            i += 1
        t = (d - acc) / segs[i] if segs[i] > 0 else 0.0
        result.append((1-t) * pts[i] + t * pts[i+1])
    return np.array(result, dtype=np.float32)

def align(ref, new):
    N = len(ref)
    best_shift, best_err = 0, 1e12
    for s in range(N):
        rolled = np.roll(new, -s, axis=0)
        err = np.sum((ref - rolled)**2)
        if err < best_err:
            best_err, best_shift = err, s
    return np.roll(new, -best_shift, axis=0)

# ---------------------------
# Point motion constraints: clamp + temporal smoothing + rigidity
# ---------------------------
def clamp_point_movement(old_pt, new_pt, max_shift_px=MAX_POINT_SHIFT):
    dx = new_pt[0] - old_pt[0]
    dy = new_pt[1] - old_pt[1]
    dist = math.hypot(dx, dy)
    if dist <= max_shift_px:
        return new_pt
    scale = max_shift_px / (dist + 1e-8)
    return (old_pt[0] + dx * scale, old_pt[1] + dy * scale)

def temporal_smooth(old_pt, new_pt, alpha=POINT_SMOOTH_ALPHA):
    # new position = old*(1-alpha) + new*alpha (alpha small => smoother)
    return (old_pt[0] * (1.0 - alpha) + new_pt[0] * alpha,
            old_pt[1] * (1.0 - alpha) + new_pt[1] * alpha)

def enforce_rigidity(pts, rigidity=SHAPE_RIGIDITY):
    # pts: Nx2 numpy array. For each point, keep it close to neighbors average
    N = len(pts)
    if N < 3 or rigidity <= 0:
        return pts
    out = pts.copy()
    for i in range(N):
        left = pts[(i-1) % N]
        right = pts[(i+1) % N]
        center = (left + right) / 2.0
        offset = out[i] - center
        # allowed offset proportional to (1-rigidity) * some scale (perimeter)
        # compute local scale from neighbor distances
        neigh_dist = (np.linalg.norm(left - pts[i]) + np.linalg.norm(right - pts[i])) / 2.0
        allowed = max(1.0, neigh_dist * (1.0 - rigidity) * 2.0)
        off_len = np.linalg.norm(offset)
        if off_len > allowed:
            out[i] = center + (offset / (off_len + 1e-8)) * allowed
    return out

# ---------------------------
# Nuke Roto creation / update helpers
# ---------------------------
def create_or_get_roto_node(name="DepthAutoRoto"):
    existing = nuke.toNode(name)
    if existing is not None:
        return existing
    roto = nuke.createNode("Roto")
    try:
        roto.setName(name)
    except Exception:
        pass
    return roto

def mask_to_shapes_info(mask, frame, z_index, is_bg, name_prefix):
    """
    Convert a binary mask into one or more contour-based shape infos.
    Returns list of dicts: { 'name', 'z_index', 'mask', 'contour', 'pts' }
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    info = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 50:
            continue
        # simplify with approx if needed
        epsilon = 1.0
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # choose number of resample pts based on perimeter
        perim = cv2.arcLength(cnt, True)
        target_pts = int(np.clip(perim / 4.0, 120, 600))
        pts = resample(np.expand_dims(approx.reshape(-1,2), axis=1), target_pts)
        pts = pts.reshape(-1,2)
        info.append({
            'name': f"{name_prefix}_{i+1:03d}",
            'z_index': z_index,
            'is_bg': is_bg,
            'mask': mask,   # original mask may be used later if needed
            'contour': cnt,
            'pts': pts,
            'frame': frame
        })
    return info

def add_shape_to_layer(curves, layer, pts, shape_name, frame, img_h):
    """
    pts: Nx2 array in image coords (x,y) with origin at top-left.
    Nuke Roto expects points in its coordinate system: bottom-left origin in many setups,
    the user's original used mask.shape[0] - y. We'll follow that convention:
    """
    shape = rp.Shape(curves)
    shape.name = shape_name
    for (x, y) in pts:
        # convert to Nuke coords: (x, height - y)
        ny = float(img_h - y)
        cp = rp.ShapeControlPoint(float(x), ny)
        shape.append(cp)
        # set an initial key - create position key at frame
        try:
            cp.center.addPositionKey(frame, (float(x), ny, 0.0))
        except Exception:
            # older/newer Nuke variants may behave differently
            try:
                cp.center.setValue((float(x), ny, 0.0))
            except Exception:
                pass
    try:
        shape.setClosed(True)
    except Exception:
        pass
    try:
        shape.setColor((1.0,0.0,0.0,1.0))
    except Exception:
        pass
    layer.append(shape)
    return shape

def z_order_layer_name(z_index):
    return f"Z{z_index:02d}"

# ---------------------------
# Main driver
# ---------------------------
def depth_auto_roto(edge_weight=EDGE_WEIGHT_DEFAULT,
                    extract_background=EXTRACT_BACKGROUND_BY_DEFAULT,
                    min_area_px=DEFAULT_MIN_AREA_PX,
                    first_frame=None, last_frame=None,
                    max_point_shift=MAX_POINT_SHIFT,
                    point_smooth_alpha=POINT_SMOOTH_ALPHA,
                    rigidity_alpha=SHAPE_RIGIDITY):
    """
    Main entry. Use with a selected Read node in Nuke.
    """
    try:
        read_node = nuke.selectedNode()
    except Exception:
        nuke.message("Select a Read node first.")
        return

    if read_node.Class() != "Read":
        nuke.message("Selected node must be a Read node.")
        return

    if first_frame is None:
        first_frame = int(read_node.firstFrame())
    if last_frame is None:
        last_frame = int(read_node.lastFrame())

    print(f"Starting Depth Auto Roto from frame {first_frame} to {last_frame}")
    roto = create_or_get_roto_node("DepthAutoRoto")
    curves = roto['curves']
    root = curves.rootLayer

    # Pre-cache device and model
    load_midas()

    # structure to hold previous masks for IoU matching
    prev_objects = []  # list of dicts: { 'name', 'mask', 'z_index', 'pts' }

    # For deterministic naming counts per z-index
    z_counts = defaultdict(int)

    # We'll create layer objects per z-index and reuse if present
    layer_map = {}  # z_index -> rp.Layer

    for frame in range(first_frame, last_frame + 1):
        t0 = time.time()
        img = read_frame_numpy(read_node, frame)
        if img is None:
            print(f"Frame {frame}: image not found, skipping")
            continue
        img_h, img_w = img.shape[:2]

        # 1) depth inference (GPU-first) - full res
        try:
            depth = infer_depth_fullres(img)
        except Exception as e:
            print("ERROR: MiDaS inference failed:", e)
            nuke.message("MiDaS model inference failed. Check torch installation and internet on first run.")
            return

        # 2) refine depth with guided filter if available, else bilateral fallback
        depth_ref = guided_refine(depth, img)

        # 3) edge maps
        d_edges = get_depth_edges(depth_ref)
        rgb_edges = get_rgb_edges(img)
        combined_edges = fuse_edges(d_edges, rgb_edges, edge_weight)

        # 4) segmentation (HARD)
        depth_layers = segment_depth_kmeans(depth_ref, hard=True)

        # if background extraction toggled off -> try to filter out deepest layers heuristically
        if not extract_background:
            # heuristic: keep layers that contain enough area / foreground presence
            # compute foreground presence defined by depth nearer than median
            med = np.median(depth_ref)
            keep = []
            for i, m in enumerate(depth_layers):
                # measure average depth over mask area
                if m.sum() == 0:
                    keep.append(False)
                    continue
                mask_depth_vals = depth_ref[m.astype(bool)]
                mean_d = mask_depth_vals.mean() if mask_depth_vals.size else 0.0
                # Keep if mean depth is closer than median or area big
                if mean_d > med or area_px(m) > (0.02 * img_w * img_h):
                    keep.append(True)
                else:
                    keep.append(False)
            # ensure we keep at least one nearest layer
            if any(keep):
                depth_layers = [m for m,k in zip(depth_layers, keep) if k]
            # else keep all (fallback)

        # 5) for each depth layer -> find connected components -> merge small ones
        frame_objects = []  # collect candidate object masks & infos
        for z_idx, layer_mask in enumerate(depth_layers):
            comps = find_components(layer_mask)
            if not comps:
                continue
            merged = merge_small_to_nearest(comps, min_area_px=min_area_px)
            for comp in merged:
                # possible refine boundaries: snap comp to combined_edges (dilate edges and flood)
                # We'll grow edges into a mask and intersect / snap using distance transform
                # but careful: keep comp as primary for now
                frame_objects.append({
                    'mask': comp,
                    'z_index': z_idx,
                })

        # 6) split or keep objects based on IoU with previous frame -> name and shape decisions
        shapes_info = []
        matched_prev = set()
        for obj in frame_objects:
            mask = obj['mask']
            z_idx = obj['z_index']
            # try match to previous objects by IoU (prefer same z_index)
            best_iou = 0.0
            best_prev = None
            for p in prev_objects:
                if p['z_index'] != z_idx:
                    continue
                val = compute_iou(p['mask'], mask)
                if val > best_iou:
                    best_iou = val
                    best_prev = p
            # decide split strategy
            if best_prev is None:
                strategy = 'single'  # new object, create single part
            else:
                if best_iou > IOU_THRESH_HIGH:
                    strategy = 'single'
                elif best_iou >= IOU_THRESH_MED:
                    strategy = 'medium'
                else:
                    strategy = 'heavy'

            # For "medium" or "heavy" attempt sub-splitting using local depth gradients and combined edges
            if strategy == 'single':
                # single shape - generate contour and pts
                name_base_idx = z_counts[z_idx] + 1
                z_counts[z_idx] = name_base_idx
                name_prefix = f"Z{z_idx:02d}_FG"
                shapes = mask_to_shapes_info(mask, frame, z_idx, False, name_prefix)
                # shapes is list - typically one; update their names to increment
                for s_i, s in enumerate(shapes):
                    s['name'] = f"{name_prefix}_{name_base_idx + s_i:03d}"
                shapes_info.extend(shapes)
            else:
                # medium/heavy: attempt segmentation inside this mask guided by edges
                # use combined_edges to seed a watershed-like split
                local_masks = split_mask_by_edges(mask, combined_edges)
                # if split small or single, fallback to original mask
                if not local_masks:
                    name_base_idx = z_counts[z_idx] + 1
                    z_counts[z_idx] = name_base_idx
                    name_prefix = f"Z{z_idx:02d}_FG"
                    shapes = mask_to_shapes_info(mask, frame, z_idx, False, name_prefix)
                    for s_i, s in enumerate(shapes):
                        s['name'] = f"{name_prefix}_{name_base_idx + s_i:03d}"
                    shapes_info.extend(shapes)
                else:
                    # merged local_masks with min_area logic
                    merged_local = merge_small_to_nearest(local_masks, min_area_px=min_area_px//2)
                    for msk in merged_local:
                        name_base_idx = z_counts[z_idx] + 1
                        z_counts[z_idx] = name_base_idx
                        name_prefix = f"Z{z_idx:02d}_FG"
                        shapes = mask_to_shapes_info(msk, frame, z_idx, False, name_prefix)
                        for s_i, s in enumerate(shapes):
                            s['name'] = f"{name_prefix}_{name_base_idx + s_i:03d}"
                        shapes_info.extend(shapes)

        # 7) Build roto (create layers for z-index if missing) and put shapes in them
        # We'll rebuild the roto layers every frame for simplicity (you may decide to do incremental updates)
        # But to preserve keyframes we will reuse existing shapes if names match by searching.

        # Create / get layers ordered near->far
        # Clear root layer children (we will repopulate per frame to ensure ordering)
        # But to preserve previous shape nodes and their animation, we try to reuse layers by name
        # For simplicity and stability, we'll create new layers under Roto for the first frame, then update shapes by name on subsequent frames

        # For simplicity here: if first frame -> create layers and shapes; subsequent frames update existing shapes via keyframing
        if frame == first_frame:
            # build fresh layers & shapes
            for s in shapes_info:
                zidx = s['z_index']
                lname = z_order_layer_name(zidx)
                if lname in layer_map:
                    layer = layer_map[lname]
                else:
                    layer = rp.Layer(curves)
                    layer.name = lname
                    root.append(layer)
                    layer_map[lname] = layer
                # create shapes inside this layer
                for s2 in [s]:
                    add_shape_to_layer(curves, layer, s2['pts'], s2['name'], frame, img_h)
        else:
            # update existing shapes: basic approach -> append shapes if not exist. If exist, move points (keyframe).
            for s in shapes_info:
                zidx = s['z_index']
                lname = z_order_layer_name(zidx)
                if lname not in layer_map:
                    # create missing layer and append
                    layer = rp.Layer(curves)
                    layer.name = lname
                    root.append(layer)
                    layer_map[lname] = layer
                    layer_map[lname] = layer
                layer = layer_map[lname]
                # attempt to find an existing shape with same name
                existing_shape = None
                for sh in layer:
                    try:
                        if sh.name == s['name']:
                            existing_shape = sh
                            break
                    except Exception:
                        continue
                if existing_shape is None:
                    # create new shape
                    add_shape_to_layer(curves, layer, s['pts'], s['name'], frame, img_h)
                else:
                    # move existing shape's control points to new positions with motion constraints
                    # build new pts in Nuke coord (x, height - y)
                    new_pts = np.array([[pt[0], img_h - pt[1]] for pt in s['pts']], dtype=np.float32)
                    # get old pts positions from shape
                    old_pts = []
                    try:
                        for cp in existing_shape:
                            # rp.ShapeControlPoint has center knob with 3-component center value
                            try:
                                cx, cy, cz = cp.center.value()
                                old_pts.append((float(cx), float(cy)))
                            except Exception:
                                # fallback, use cp.x and cp.y if available
                                try:
                                    old_pts.append((float(cp.x), float(cp.y)))
                                except Exception:
                                    old_pts.append((0.0, 0.0))
                    except Exception:
                        # no old pts accessable, fallback to new
                        old_pts = [(float(x), float(y)) for x,y in new_pts]

                    # if lengths mismatch, re-sample new_pts to match old count
                    if len(old_pts) != len(new_pts):
                        # resample to old count
                        N = max(3, len(old_pts))
                        # simple linear resample
                        inds = np.linspace(0, len(new_pts)-1, N).astype(int)
                        new_pts = new_pts[inds]
                    # apply clamp, smoothing, rigidity
                    old_arr = np.array(old_pts, dtype=np.float32)
                    new_arr = np.array(new_pts, dtype=np.float32)
                    processed = []
                    for oi, (o, n) in enumerate(zip(old_arr, new_arr)):
                        clamped = clamp_point_movement((o[0], o[1]), (n[0], n[1]), max_shift_px=max_point_shift)
                        sm = temporal_smooth((o[0], o[1]), clamped, alpha=point_smooth_alpha)
                        processed.append(sm)
                    processed = np.array(processed, dtype=np.float32)
                    processed = enforce_rigidity(processed, rigidity=rigidity_alpha)
                    # write back to control points (keyframe)
                    for cp, (x, y) in zip(existing_shape, processed):
                        try:
                            # convert back to original coordinate system used in move_point
                            # The earlier code used cp.center.addPositionKey(frame, (x, y, 0))
                            cp.center.addPositionKey(frame, (float(x), float(y), 0.0))
                        except Exception as e:
                            try:
                                cp.center.setValue((float(x), float(y), 0.0))
                            except Exception:
                                pass

        curves.changed()
        t1 = time.time()
        print(f"Frame {frame} processed in {t1 - t0:.2f}s, created {len(shapes_info)} shapes")
        prev_objects = [{'mask': s['mask'], 'z_index': s['z_index']} for s in shapes_info]

    try:
        nuke.show(roto)
    except Exception:
        pass
    nuke.message("✅ Depth-assisted auto-roto finished.")

# ---------------------------
# Small helper functions used above (must be below in file)
# ---------------------------
def compute_iou(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return 0.0
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def mask_to_shapes_info(mask, frame, z_index, is_bg, name_prefix):
    # Reuse the earlier function (placed again to ensure scoping)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    info = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 50:
            continue
        approx = cv2.approxPolyDP(cnt, 0.5, True)
        perim = cv2.arcLength(cnt, True)
        target_pts = int(np.clip(perim / 4.0, 120, 600))
        pts = resample(np.expand_dims(approx.reshape(-1,2), axis=1), target_pts).reshape(-1,2)
        info.append({
            'name': f"{name_prefix}_{i+1:03d}",
            'z_index': z_index,
            'is_bg': is_bg,
            'mask': mask,
            'contour': cnt,
            'pts': pts,
            'frame': frame
        })
    return info

def split_mask_by_edges(mask, combined_edges):
    """
    Attempt to split a single mask into sub-masks using combined_edges within that mask.
    We'll perform watershed-like seeding: peaks in edge map define boundaries; invert edges as distance map.
    Return list of masks or [] if no good split.
    """
    try:
        # focus on local region bounding box
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        x0, x1 = max(0, xs.min()-5), min(mask.shape[1]-1, xs.max()+5)
        y0, y1 = max(0, ys.min()-5), min(mask.shape[0]-1, ys.max()+5)
        sub_mask = mask[y0:y1+1, x0:x1+1]
        sub_edges = combined_edges[y0:y1+1, x0:x1+1]
        # threshold edges to seeds
        # invert edges to get markers for watershed
        inv = (1.0 - sub_edges)
        inv8 = (inv * 255.0).astype(np.uint8)
        # distance transform on inverted edges masked by sub_mask
        dist = cv2.distanceTransform((sub_mask[y0:y1+1, x0:x1+1]//255).astype(np.uint8), cv2.DIST_L2, 5)
        # markers: local maxima in distance
        ret, markers = cv2.threshold(dist, dist.mean() * 0.6, 255, cv2.THRESH_BINARY)
        markers = markers.astype(np.uint8)
        num_labels, labs = cv2.connectedComponents(markers, connectivity=8)
        if num_labels <= 1:
            return []
        # use watershed on a 3-channel image to get segments
        sub_rgb = cv2.cvtColor(inv8, cv2.COLOR_GRAY2BGR)
        labs32 = labs.astype(np.int32)
        cv2.watershed(sub_rgb, labs32)
        masks = []
        for lab in range(1, np.max(labs32)+1):
            comp = (labs32 == lab).astype(np.uint8)
            # apply original sub_mask
            comp = cv2.bitwise_and(comp, (sub_mask//255).astype(np.uint8))
            if comp.sum() < 50:
                continue
            # convert to full-frame mask
            full = np.zeros_like(mask)
            full[y0:y1+1, x0:x1+1] = comp * 255
            masks.append(full)
        return masks
    except Exception as e:
        # fallback: no split
        return []

# ---------------------------
# expose callable from UI/script editor
# ---------------------------
def run_depth_auto_roto_with_defaults():
    depth_auto_roto(edge_weight=EDGE_WEIGHT_DEFAULT,
                    extract_background=EXTRACT_BACKGROUND_BY_DEFAULT,
                    min_area_px=DEFAULT_MIN_AREA_PX,
                    max_point_shift=MAX_POINT_SHIFT,
                    point_smooth_alpha=POINT_SMOOTH_ALPHA,
                    rigidity_alpha=SHAPE_RIGIDITY)

# If you want to run immediately, uncomment:
# run_depth_auto_roto_with_defaults()

# End of script
