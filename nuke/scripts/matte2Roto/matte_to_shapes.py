#Currently this version only works with png
print("check point")
import nuke
import nuke.rotopaint as rp
import cv2
import numpy as np
import os
import re

# ---------------- Configuration ----------------
# Toggle to control how dense the roto shapes should be
MAX_DETAIL = True     # Set True for very high fidelity (slower)
# ------------------------------------------------


def get_seq_filename(node, frame):
    """Resolve frame file path properly even if #### or %04d pattern is used."""
    file_knob = node["file"].value()
    if "####" in file_knob:
        file_path = file_knob.replace("####", f"{frame:04d}")
    elif re.search(r"%0\dd", file_knob):
        width = int(re.search(r"%0(\d)d", file_knob).group(1))
        file_path = re.sub(r"%0\dd", f"{frame:0{width}d}", file_knob)
    else:
        file_path = file_knob
    return file_path


def read_mask(node, frame):
    path = get_seq_filename(node, frame)
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # auto invert if necessary
    if np.sum(mask == 0) < np.sum(mask == 255):
        mask = cv2.bitwise_not(mask)
    return mask


def resample(contour, target):
    pts = contour[:, 0, :].astype(np.float32)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    segs = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
    L = np.sum(segs)
    if L < 1e-6:
        return np.repeat(pts[:1], target, axis=0)

    dists = np.linspace(0, L, target, endpoint=False)
    result, acc, i = [], 0.0, 0
    for d in dists:
        while i < len(segs) - 1 and acc + segs[i] < d:
            acc += segs[i]
            i += 1
        t = (d - acc) / segs[i] if segs[i] > 0 else 0
        result.append((1 - t) * pts[i] + t * pts[i + 1])
    return np.array(result, dtype=np.float32)


def align(ref, new):
    N = len(ref)
    best_shift, best_err = 0, 1e12
    for s in range(N):
        rolled = np.roll(new, -s, axis=0)
        err = np.sum((ref - rolled) ** 2)
        if err < best_err:
            best_err, best_shift = err, s
    return np.roll(new, -best_shift, axis=0)


def move_point(cp, frame, x, y):
    """Keyframe the control point position for Nuke 16+."""
    try:
        cp.center.addPositionKey(frame, (x, y, 0))
    except Exception as e:
        print(f"⚠️ Could not keyframe at frame {frame}: {e}")


def matte_to_shapes_animated_from_sequence():
    try:
        node = nuke.selectedNode()
    except:
        nuke.message("Please select a Read node first.")
        return

    if node.Class() != "Read":
        nuke.message("Selected node must be a Read node.")
        return

    first = int(node.firstFrame())
    last = int(node.lastFrame())
    print(f"Processing from frame {first} to {last}")

    # --- Gather all contours ---
    contour_data = {}
    for frame in range(first, last + 1):
        mask = read_mask(node, frame)
        if mask is None:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour_data[frame] = contours

    if not contour_data:
        nuke.message("No contour data found in sequence.")
        return

    # --- Create Roto node and initial shapes from first frame ---
    roto = nuke.createNode("Roto")
    roto["format"].setValue(node["format"].value())
    curves = roto["curves"]
    root = curves.rootLayer
    layer = rp.Layer(curves)
    layer.name = "Generated_Mattes"
    root.append(layer)

    first_frame_contours = contour_data[first]
    h = list(contour_data.values())[0][0].shape[0]
    shapes = []

    for i, cnt in enumerate(first_frame_contours):
        if cv2.contourArea(cnt) < 50:
            continue

        # --- Adaptive contour sampling for smoother shapes ---
        epsilon = 2.0 if MAX_DETAIL else 1.0
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2)

        #compute perimeter of simplified contour
        perimeter = cv2.arcLength(approx, True)

        #commenting out max_detail logic
        # if MAX_DETAIL:
        #     target_points = int(np.clip(perimeter / 5.0, 100, 150))
        # else:
        #     target_points = int(np.clip(perimeter / 8.0, 60, 120))

        target_points = int(np.clip(perimeter / 5.0, 100, 150))


        pts = resample(np.expand_dims(pts, axis=1), target_points)
        pts = pts.reshape(-1, 2)
        # -------------------------------------------------------

        shape = rp.Shape(curves)
        shape.name = f"TrackedShape_{i+1}"

        for (x, y) in pts:
            cp = rp.ShapeControlPoint(float(x), float(mask.shape[0] - y))
            shape.append(cp)
            # ensure first keyframe exists
            cp.center.addPositionKey(first, (x, y, 0))

        try:
            shape.setClosed(True)
        except:
            pass
        try:
            shape.setColor((1.0, 0.0, 0.0, 1.0))
        except:
            pass

        layer.append(shape)
        print("Shape", shape.name, "point", len(shape)) #debug statement to print points
        shapes.append((shape, pts.copy()))

    # --- Stabilize contour data to avoid shape mismatch ---
    max_shapes = len(shapes)
    for f, cts in contour_data.items():
        if len(cts) < max_shapes:
            cts += [cts[-1]] * (max_shapes - len(cts))
        contour_data[f] = sorted(cts, key=cv2.contourArea, reverse=True)
    # -------------------------------------------------------

    # --- Animate shapes using later frames ---
    for frame in range(first + 1, last + 1):
        contours = contour_data.get(frame)
        if not contours:
            continue

        for (shape, ref_pts), cnt in zip(shapes, contours):
            new = resample(cnt, len(ref_pts))
            new = align(ref_pts, new)

            for cp, (x, y) in zip(shape, new):
                move_point(cp, frame, float(x), float(mask.shape[0] - y))

            ref_pts[:] = new

    curves.changed()
    nuke.show(roto)
    nuke.message("✅ Animated roto created successfully from matte sequence ✅")


def run():
    matte_to_shapes_animated_from_sequence()

run()
