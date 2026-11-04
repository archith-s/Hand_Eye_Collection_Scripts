#!/usr/bin/env python3

# Measures how accurate the "ground truth" gripper pose -- the hand-eye
# calibration with hand_eye_rodrigues_dataset.py's / 3d_pose_alignment.py's
# GUI alignment folded in (the calibration file's base-frame.transform),
# composed with PSM kinematics -- really is, against hand-drawn
# segmentations of the wrist clevis. Meant to be run AFTER
# hand_eye_rodrigues_dataset.py has applied its correction to the
# hand-eye calibration file.
#
# Three phases, run back-to-back in one sitting by default:
#
#   1. Record: only what the evaluation needs -- the stereo-rectified right
#      image plus the GT gripper / shaft-end poses, computed exactly like
#      image_generation_gt_hand_eye.py (whose calibration/rectification
#      helpers are reused). 'c' to capture, 'q'/ESC to finish.
#
#   2. Segment: draw a mask of the CLEVIS (the wrist link between the
#      pitch pin and the jaw pin) on each right_rect frame. The window shows
#      two panels sharing one zoom/pan: the image to segment on the left,
#      the same image with the GT clevis silhouette rendered over it on the
#      right. Polygon tool by default (click vertices, Enter/space to
#      close), brush with 'b'. Masks go to masks/mask-XXXX.png.
#
#   3. Evaluate: the clevis CAD model (SurgRIPE joint.stl, posed from
#      cam_T_shaft_end + cam_T_gripper exactly like
#      PositionOpitmizationDiffRendering's hand_eye_diff_render_test.py)
#      is rendered at the ground-truth pose and compared to the drawn mask:
#
#        - IoU between the GT clevis silhouette and the drawn mask
#        - rotational / translational error: a 6-DoF rigid correction is
#          fitted so the rendered clevis best matches the drawn mask, and
#          the error is GT clevis pose vs that fitted pose (translation at
#          the pitch pin, camera frame). A silhouette constrains depth
#          (camera z) far more weakly than x/y, so translation is also
#          split into in-plane and depth.
#
#      Results go to segmentation_eval.pdf in the dataset folder: a summary
#      table, then one page per frame overlaying the GT clevis silhouette on
#      the segmentation (what the IoU is) and the GT vs best-fit outlines
#      (what the errors are).
#
# Once the PDF is written and every frame is segmented or skipped, the
# images, masks and intermediate JSON are deleted, leaving only the PDF
# (for a dataset recorded by another script, only the masks are deleted).
#
# --annotate-only skips phase 1 (segment + evaluate an existing dataset,
# e.g. one recorded by image_generation_gt_hand_eye.py), --evaluate-only
# skips 1 and 2. Neither needs ROS/dVRK.

import argparse
import json
import os
import shutil
import struct
import sys
import time

import cv2
import numpy
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

DEFAULT_HAND_EYE_JSON = os.path.expanduser('~/PSM1-registration-open-cv.json')
DEFAULT_CLEVIS_STL = os.path.expanduser('~/TEST/joint.stl')

GT_JSON_NAME = 'gripper_ground_truth.json'
RECORDED_BY = 'gt_segmentation_eval.py'
ANNOTATIONS_JSON_NAME = 'segmentation_annotations.json'
MASK_SUBDIR = 'masks'
EVAL_PDF_NAME = 'segmentation_eval.pdf'

REGION = 'clevis'
INSTRUCTION = ('Draw the CLEVIS only: the wrist link from the first (pitch) pin to the jaw pin. '
               'NOT the shaft, NOT the jaws.')

# pose fit: (proxy-mesh cell size mm, blur sigma in fit px, initial simplex
# step (mm, deg), max function evaluations), coarse to fine
FIT_STAGES = [
    (0.30, 3.0, (2.0, 4.0), 300),
    (0.10, 1.5, (0.7, 1.5), 300),
    (0.05, 0.7, (0.25, 0.5), 300),
]
FIT_MAX_SIDE_PX = 320      # fit crop is downscaled so its longest side is at most this
FIT_SUPERSAMPLE = 2        # render at 2x then area-downsample, for a smoother objective
FIT_RESTART_RANGE = (3.0, 5.0)  # +-mm, +-deg uniform perturbation of the extra starts

LIVE_WINDOW_NAME = 'Clevis capture (c: capture, q: finish)'
ANNOTATE_WINDOW_NAME = 'Segment clevis (Enter: close polygon, s: save+next, q: quit -- keys in terminal)'
MASK_COLOR = (0, 0, 255)
GT_COLOR = (0, 255, 0)
HISTORY_LIMIT = 30
ANNOTATE_WINDOW_WIDTH = 1800  # initial on-screen width of the two-panel window
SEPARATOR_PX = 6


# ---------------------------------------------------------------------------
# clevis model
# ---------------------------------------------------------------------------

def load_stl(path: str):
    with open(path, 'rb') as f:
        data = f.read()
    count = struct.unpack('<I', data[80:84])[0] if len(data) >= 84 else 0
    if 84 + 50 * count == len(data):
        dtype = numpy.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
        triangles = numpy.frombuffer(data[84:], dtype=dtype)['v'].astype(numpy.float64)
    else:  # ASCII
        coords = [line.split()[1:4] for line in data.decode(errors='ignore').splitlines()
                  if line.strip().startswith('vertex')]
        triangles = numpy.array(coords, dtype=numpy.float64).reshape(-1, 3, 3)
    # weld the per-triangle vertices
    vertices, inverse = numpy.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, inverse.ravel().reshape(-1, 3)


def cluster_decimate(vertices: numpy.ndarray, faces: numpy.ndarray, cell: float):
    # Vertex clustering, keeping one ORIGINAL vertex per cell (not the cell
    # mean, which would shrink the silhouette by up to cell/2 everywhere and
    # bias the fitted depth).
    keys = numpy.floor(vertices / cell).astype(numpy.int64)
    _, first, inverse = numpy.unique(keys, axis=0, return_index=True, return_inverse=True)
    new_faces = inverse.ravel()[faces]
    keep = ((new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 0] != new_faces[:, 2]))
    new_faces = numpy.unique(numpy.sort(new_faces[keep], axis=1), axis=0)
    return vertices[first], new_faces


def cam_T_joint_model(cam_T_shaft: numpy.ndarray, cam_T_gripper: numpy.ndarray) -> numpy.ndarray:
    # Wrist-clevis pose of the SurgRIPE joint.stl mesh from both dVRK axes --
    # copied from hand_eye_diff_render_test.py, see its docstring: pitch pin
    # at the mesh origin along X, yaw pin 9.1mm forward, body along -Y.
    shaft_origin = cam_T_shaft[:3, 3]
    forward = cam_T_gripper[:3, 3] - shaft_origin
    forward = forward / numpy.linalg.norm(forward)
    z_axis = cam_T_gripper[:3, 0] - forward * numpy.dot(cam_T_gripper[:3, 0], forward)
    z_axis = z_axis / numpy.linalg.norm(z_axis)
    y_axis = -forward
    x_axis = numpy.cross(y_axis, z_axis)
    matrix = numpy.eye(4)
    matrix[:3, :3] = numpy.column_stack([x_axis, y_axis, z_axis])
    matrix[:3, 3] = shaft_origin
    return matrix


class ClevisModel:
    # vertices in meters in the clevis (joint.stl) frame
    def __init__(self, stl_path: str):
        v, f = load_stl(stl_path)
        if numpy.abs(v).max() >= 1.0:  # mm -> m
            v = v / 1000.0
        self.vertices, self.faces = v, f
        self.source = os.path.abspath(stl_path)
        self._proxies = {}
        print(f'Clevis: {stl_path} ({len(f)} faces)')

    def level(self, cell_mm: float = None):
        if cell_mm is None:
            return self.vertices, self.faces
        if cell_mm not in self._proxies:
            self._proxies[cell_mm] = cluster_decimate(self.vertices, self.faces, cell_mm / 1000.0)
        return self._proxies[cell_mm]

    def posed(self, cam_T_clevis, cell_mm: float = None):
        v, f = self.level(cell_mm)
        return v @ cam_T_clevis[:3, :3].T + cam_T_clevis[:3, 3], f


# ---------------------------------------------------------------------------
# silhouette rendering
# ---------------------------------------------------------------------------

def window_camera(K: numpy.ndarray, x0: float, y0: float, scale: float) -> numpy.ndarray:
    # intrinsics for the window starting at full-image pixel (x0, y0),
    # resampled by `scale` (pixel centers preserved)
    Kw = K.copy()
    Kw[0, 0] *= scale
    Kw[1, 1] *= scale
    Kw[0, 2] = (K[0, 2] + 0.5 - x0) * scale - 0.5
    Kw[1, 2] = (K[1, 2] + 0.5 - y0) * scale - 0.5
    return Kw


def render_silhouette(model: ClevisModel, cam_T_clevis, K, size, cell_mm=None, supersample=1) -> numpy.ndarray:
    # A mesh's silhouette is just the union of its projected triangles, so
    # no z-buffer is needed. Each triangle is filled on its own: cv2.fillPoly
    # with several polygons uses even-odd filling, which punches holes
    # wherever triangles overlap. Returns uint8 coverage (0-255).
    w, h = size
    Ks = window_camera(K, 0.0, 0.0, supersample) if supersample > 1 else K
    W, H = w * supersample, h * supersample
    mask = numpy.zeros((H, W), numpy.uint8)
    points, faces = model.posed(cam_T_clevis, cell_mm)
    faces = faces[(points[:, 2] > 1e-4)[faces].all(axis=1)]
    if len(faces):
        uvw = points @ Ks.T
        uv = numpy.clip(uvw[:, :2] / numpy.maximum(uvw[:, 2:3], 1e-9), -1e5, 1e5)
        tris = uv[faces]
        on_image = ((tris[:, :, 0].max(1) >= 0) & (tris[:, :, 0].min(1) < W)
                    & (tris[:, :, 1].max(1) >= 0) & (tris[:, :, 1].min(1) < H))
        for tri in numpy.round(tris[on_image] * 16).astype(numpy.int32):
            cv2.fillConvexPoly(mask, tri, 255, cv2.LINE_8, 4)
    if supersample > 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    return mask


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def iou(a: numpy.ndarray, b: numpy.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def soft_iou(a: numpy.ndarray, b: numpy.ndarray) -> float:
    union = numpy.maximum(a, b).sum()
    return float(numpy.minimum(a, b).sum() / union) if union > 0 else 0.0


def delta_from_params(x, pivot) -> numpy.ndarray:
    # x = (tx, ty, tz in mm, rotation vector in degrees), applied in the
    # camera frame about `pivot` (the clevis centroid) so rotation doesn't
    # drag the clevis sideways
    R = Rotation.from_rotvec(numpy.radians(x[3:6])).as_matrix()
    delta = numpy.eye(4)
    delta[:3, :3] = R
    delta[:3, 3] = (numpy.eye(3) - R) @ pivot + numpy.asarray(x[0:3]) / 1000.0
    return delta


def fit_pose_to_mask(model: ClevisModel, K, drawn: numpy.ndarray, cam_T_clevis, window,
                     restarts: int, rng: numpy.random.Generator) -> numpy.ndarray:
    # Returns the camera-frame clevis pose whose silhouette best matches the
    # drawn mask: Nelder-Mead on soft IoU, coarse to fine, from the GT pose
    # plus restarts-1 perturbed starts; the start with the best full-mesh IoU wins.
    x0, y0, x1, y1 = window
    scale = min(1.0, FIT_MAX_SIDE_PX / max(x1 - x0, y1 - y0))
    fw, fh = max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))
    Kf = window_camera(K, x0, y0, scale)
    Kfull = window_camera(K, x0, y0, 1.0)
    target = cv2.resize(drawn[y0:y1, x0:x1].astype(numpy.float32), (fw, fh), interpolation=cv2.INTER_AREA)
    drawn_window = drawn[y0:y1, x0:x1]
    pivot = model.posed(cam_T_clevis)[0].mean(axis=0)

    def render_fit(x, cell_mm):
        return render_silhouette(model, delta_from_params(x, pivot) @ cam_T_clevis, Kf, (fw, fh), cell_mm,
                                 FIT_SUPERSAMPLE).astype(numpy.float32) / 255.0

    starts = [numpy.zeros(6)]
    t_rng, r_rng = FIT_RESTART_RANGE
    for _ in range(max(0, restarts - 1)):
        starts.append(numpy.concatenate([rng.uniform(-t_rng, t_rng, 3), rng.uniform(-r_rng, r_rng, 3)]))

    best_iou, best_pose = -1.0, cam_T_clevis
    for start in starts:
        x = start
        for cell_mm, sigma, (t_step, r_step), max_evals in FIT_STAGES:
            target_blur = cv2.GaussianBlur(target, (0, 0), sigma)
            simplex = numpy.vstack([x] + [x + numpy.eye(6)[i] * (t_step if i < 3 else r_step) for i in range(6)])
            x = minimize(
                lambda p: 1.0 - soft_iou(cv2.GaussianBlur(render_fit(p, cell_mm), (0, 0), sigma), target_blur),
                x, method='Nelder-Mead',
                options={'initial_simplex': simplex, 'maxfev': max_evals, 'xatol': 0.01, 'fatol': 1e-5}).x
        pose = delta_from_params(x, pivot) @ cam_T_clevis
        score = iou(render_silhouette(model, pose, Kfull, (x1 - x0, y1 - y0)) > 127, drawn_window)
        if score > best_iou:
            best_iou, best_pose = score, pose
    return best_pose


def pose_error(cam_T_gt: numpy.ndarray, cam_T_fit: numpy.ndarray) -> dict:
    rot = Rotation.from_matrix(cam_T_fit[:3, :3] @ cam_T_gt[:3, :3].T)
    dt_mm = (cam_T_fit[:3, 3] - cam_T_gt[:3, 3]) * 1000.0
    return {
        'rot_err_deg': float(numpy.degrees(rot.magnitude())),
        'trans_err_mm': float(numpy.linalg.norm(dt_mm)),
        'trans_err_inplane_mm': float(numpy.linalg.norm(dt_mm[:2])),
        'trans_err_depth_mm': float(abs(dt_mm[2])),
        'dxyz_mm': dt_mm.tolist(),
    }


# ---------------------------------------------------------------------------
# dataset / annotation persistence
# ---------------------------------------------------------------------------

def load_gt_records(dataset_dir: str) -> list:
    path = os.path.join(dataset_dir, GT_JSON_NAME)
    if not os.path.exists(path):
        raise SystemExit(f'{path} not found -- record a dataset first (without --annotate-only/--evaluate-only)')
    with open(path, 'r') as f:
        records = json.load(f)
    if not records:
        raise SystemExit(f'{path} has no recorded frames')
    if 'camera_matrix_right_rect' not in records[0] or 'cam_T_shaft_end' not in records[0]:
        raise SystemExit(f'{path} was recorded by an older image_generation_gt_hand_eye.py (no stereo-rectified '
                         f'intrinsics / shaft-end pose) -- re-record it.')
    return records


def load_annotations(dataset_dir: str) -> dict:
    path = os.path.join(dataset_dir, ANNOTATIONS_JSON_NAME)
    annotations = {'meta': {}, 'frames': {}}
    if os.path.exists(path):
        with open(path, 'r') as f:
            annotations = json.load(f)
    region = annotations['meta'].get('region')
    if annotations['frames'] and region != REGION:
        raise SystemExit(f'{path} holds masks of a different region ({region or annotations["meta"]}), not the '
                         f'clevis -- move it and {MASK_SUBDIR}/ aside and segment again.')
    annotations['meta'].update({'region': REGION, 'instruction': INSTRUCTION})
    return annotations


def save_annotations(dataset_dir: str, annotations: dict):
    with open(os.path.join(dataset_dir, ANNOTATIONS_JSON_NAME), 'w') as f:
        json.dump(annotations, f, indent=2)


def cleanup(dataset_dir: str, records: list, annotations: dict):
    frames = annotations['frames']
    pending = [r['frame_index'] for r in records
               if frames.get(str(r['frame_index']), {}).get('status') not in ('saved', 'skipped')]
    if pending:
        print(f'Frame(s) {pending} not segmented yet -- keeping the images; finish with --annotate-only '
              f'and they are removed after that run.')
        return
    shutil.rmtree(os.path.join(dataset_dir, MASK_SUBDIR), ignore_errors=True)
    os.remove(os.path.join(dataset_dir, ANNOTATIONS_JSON_NAME))
    if all(r.get('recorded_by') == RECORDED_BY for r in records):
        for r in records:
            path = os.path.join(dataset_dir, r['right_rect_image'])
            if os.path.exists(path):
                os.remove(path)
        os.remove(os.path.join(dataset_dir, GT_JSON_NAME))
        print(f'Removed the images, masks and intermediate JSON -- only {EVAL_PDF_NAME} is left.')
    else:
        print('Removed the masks. This dataset was recorded by another script, so its own images were left.')


# ---------------------------------------------------------------------------
# phase 1: record
# ---------------------------------------------------------------------------

def run_record(args) -> bool:
    # imported here so segment/evaluate work without ROS
    import crtk
    import cv_bridge
    import dvrk_camera_registration
    import sensor_msgs.msg
    import image_generation_gt_hand_eye as gt

    calibration = gt.load_calibration_variants(args.hand_eye_json)
    if not calibration['alignment_applied']:
        print(f'\nWARNING: {args.hand_eye_json} has no applied GUI alignment (base-frame == pre-alignment), so '
              f'this would evaluate the RAW hand-eye calibration, not hand-eye + 3D pose alignment. Did you '
              f'answer "y" to applying the correction at the end of hand_eye_rodrigues_dataset.py?')
        if input('Record anyway? [y/N]: ').strip().lower() != 'y':
            return False

    rectification = gt.load_stereo_rectification(args.stereo_calib_dir or gt.DEFAULT_STEREO_CALIB_DIR)
    # hand-eye is relative to the raw right optical frame; the images are
    # stereo-rectified, so poses go through rect_T_cam_raw and project with
    # the rectified P's 3x3 (same as image_generation_gt_hand_eye.py)
    cam_T_base = calibration['post']
    rect_T_cam_raw = rectification['rect_T_cam_raw']
    camera_matrix = rectification['camera_matrix_right_rect']

    os.makedirs(args.output_dir, exist_ok=True)
    gt_path = os.path.join(args.output_dir, GT_JSON_NAME)
    records = []
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            records = json.load(f)

    ral = crtk.ral('gt_segmentation_eval')
    arm_handle = dvrk_camera_registration.ARM(ral, arm_name=args.psm_name, expected_interval=0.5)
    ral.spin()
    ral.check_connections()

    bridge = cv_bridge.CvBridge()
    latest = {}
    image_topic = f'/{gt.RIG}/right/image_raw'
    print(f'Subscribing to {image_topic}')
    ral._node.create_subscription(sensor_msgs.msg.Image, image_topic,
                                  lambda msg: latest.__setitem__('right', msg), 10)
    start_time = time.time()

    def capture():
        if 'right' not in latest:
            print('No image received yet.')
            return
        if time.time() - start_time < args.startup_delay:
            print('Still within startup delay.')
            return
        try:
            gripper_cp, _ = arm_handle.local.measured_cp()
            jp, _ = arm_handle.measured_jp()
            shaft_end_cp = arm_handle.local.forward_kinematics(jp[0:4])
        except TimeoutError as e:
            print(f'Cannot capture, PSM pose not available: {e}')
            return
        image = gt.rectify(rectification, 'right', bridge.imgmsg_to_cv2(latest['right'], desired_encoding='bgr8'))
        idx = len(records)
        fname = f'right_rect-{idx:04d}.png'
        cv2.imwrite(os.path.join(args.output_dir, fname), image)
        records.append({
            'frame_index': idx,
            'right_rect_image': fname,
            'cam_T_gripper': (rect_T_cam_raw @ cam_T_base @ gt.measured_cp_to_matrix(gripper_cp)).tolist(),
            'cam_T_shaft_end': (rect_T_cam_raw @ cam_T_base @ gt.measured_cp_to_matrix(shaft_end_cp)).tolist(),
            'camera_matrix_right_rect': camera_matrix.tolist(),
            'hand_eye_json': os.path.abspath(args.hand_eye_json),
            'alignment_applied': calibration['alignment_applied'],
            'recorded_by': RECORDED_BY,
        })
        with open(gt_path, 'w') as f:
            json.dump(records, f, indent=2)
        print(f'Captured frame {idx}')

    print('Press c to capture a frame (keep the clevis clearly visible, vary pose/depth/roll), q or ESC to finish.')
    cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while not ral.is_shutdown():
            if 'right' in latest:
                img = gt.rectify(rectification, 'right',
                                 bridge.imgmsg_to_cv2(latest['right'], desired_encoding='bgr8'))
                line = f'Captured {len(records)}  --  [c]apture  [q/ESC] finish'
                cv2.putText(img, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(LIVE_WINDOW_NAME, img)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('c'):
                capture()
    except KeyboardInterrupt:
        pass
    cv2.destroyWindow(LIVE_WINDOW_NAME)
    print(f'Captured {len(records)} frame(s) in ./{args.output_dir}/')
    ral.shutdown()
    return len(records) > 0


# ---------------------------------------------------------------------------
# phase 2: segment
# ---------------------------------------------------------------------------

class Annotator:
    def __init__(self, dataset_dir: str, records: list, annotations: dict, model: ClevisModel,
                 reference=None):
        # reference: optional callable record -> (uint8 0/1 mask, label) shown in the right panel instead of
        # the GT clevis silhouette (e.g. an automatically generated mask to compare the drawing with)
        self.dataset_dir = dataset_dir
        self.records = records
        self.annotations = annotations
        self.model = model
        self.reference = reference
        os.makedirs(os.path.join(dataset_dir, MASK_SUBDIR), exist_ok=True)

        self.mode = 'polygon'
        self.erase = False
        self.brush_radius = 6
        self.show_mask = True
        self.zoom = 1.0
        self.view_x0 = 0.0
        self.view_y0 = 0.0
        self.cursor = (0, 0)
        self.polygon = []
        self._painting = False
        self._last_paint = None
        self._pan_from = None
        self._redraw = True

        start = next((i for i, r in enumerate(records)
                      if self._entry(r).get('status', 'pending') == 'pending'), 0)
        cv2.namedWindow(ANNOTATE_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(ANNOTATE_WINDOW_NAME, self._on_mouse)
        self.index = -1
        self._load(start)
        h, w = self.image.shape[:2]
        cv2.resizeWindow(ANNOTATE_WINDOW_NAME, ANNOTATE_WINDOW_WIDTH, int(ANNOTATE_WINDOW_WIDTH * h / (2 * w)))

    # -- frames -----------------------------------------------------------

    def _entry(self, record) -> dict:
        return self.annotations['frames'].get(str(record['frame_index']), {})

    def _load(self, index):
        self.index = index
        record = self.records[index]
        self.image = cv2.imread(os.path.join(self.dataset_dir, record['right_rect_image']))
        if self.image is None:
            raise SystemExit(f'Could not read {record["right_rect_image"]}')
        h, w = self.image.shape[:2]
        entry = self._entry(record)
        mask = None
        if entry.get('mask'):
            mask = cv2.imread(os.path.join(self.dataset_dir, entry['mask']), cv2.IMREAD_GRAYSCALE)
        self.mask = (mask > 127).astype(numpy.uint8) * 255 if mask is not None else numpy.zeros((h, w), numpy.uint8)
        if self.reference is not None:
            ref, self.reference_label = self.reference(record)
            self.gt_mask = (numpy.asarray(ref) > 0).astype(numpy.uint8)
        else:
            cam_T_clevis = cam_T_joint_model(numpy.array(record['cam_T_shaft_end'], dtype=numpy.float64),
                                             numpy.array(record['cam_T_gripper'], dtype=numpy.float64))
            K = numpy.array(record['camera_matrix_right_rect'], dtype=numpy.float64)
            self.gt_mask = (render_silhouette(self.model, cam_T_clevis, K, (w, h)) > 127).astype(numpy.uint8)
            self.reference_label = 'Ground truth: hand-eye + 3D pose alignment (clevis silhouette)'
        self.history = []
        self.polygon = []
        self.dirty = False
        self.loaded_at = time.time()
        self._redraw = True

    def _save(self, status='saved'):
        record = self.records[self.index]
        key = str(record['frame_index'])
        entry = dict(self.annotations['frames'].get(key, {}))
        if status == 'saved':
            rel = os.path.join(MASK_SUBDIR, f'mask-{record["frame_index"]:04d}.png')
            cv2.imwrite(os.path.join(self.dataset_dir, rel), self.mask)
            entry['mask'] = rel
            entry['area_px'] = int((self.mask > 0).sum())
        entry['status'] = status
        entry['image'] = record['right_rect_image']
        entry['updated_at'] = time.time()
        entry['edit_seconds'] = entry.get('edit_seconds', 0.0) + (time.time() - self.loaded_at)
        self.loaded_at = time.time()
        self.annotations['frames'][key] = entry
        save_annotations(self.dataset_dir, self.annotations)
        self.dirty = False
        print(f'{status.capitalize()} frame {record["frame_index"]}'
              + (f' ({entry["area_px"]} px)' if status == 'saved' else ''))

    def _save_if_dirty(self):
        if not self.dirty:
            return
        if self.mask.any():
            self._save()
        else:
            print(f'Frame {self.records[self.index]["frame_index"]}: mask is empty, not saved (x to skip it).')

    def _go(self, delta):
        self._save_if_dirty()
        self._load(max(0, min(len(self.records) - 1, self.index + delta)))

    # -- editing ------------------------------------------------------------

    def _push_history(self):
        self.history.append(self.mask.copy())
        del self.history[:-HISTORY_LIMIT]
        self.dirty = True

    def _close_polygon(self):
        if len(self.polygon) < 3:
            print('A polygon needs at least 3 points.')
            return
        self._push_history()
        pts = numpy.round(numpy.array(self.polygon) * 16).astype(numpy.int32)
        cv2.fillPoly(self.mask, [pts], 0 if self.erase else 255, cv2.LINE_8, 4)
        self.polygon = []

    def _paint(self, p0, p1):
        value = 0 if self.erase else 255
        a = tuple(int(round(c)) for c in p0)
        b = tuple(int(round(c)) for c in p1)
        cv2.line(self.mask, a, b, value, 2 * self.brush_radius + 1)
        cv2.circle(self.mask, b, self.brush_radius, value, -1)

    # -- view ---------------------------------------------------------------

    def _to_image(self, cx, cy):
        return self.view_x0 + cx / self.zoom, self.view_y0 + cy / self.zoom

    def _clamp_view(self):
        h, w = self.image.shape[:2]
        self.view_x0 = min(max(self.view_x0, 0.0), w - w / self.zoom)
        self.view_y0 = min(max(self.view_y0, 0.0), h - h / self.zoom)

    def _zoom_at(self, cx, cy, factor):
        ix, iy = self._to_image(cx, cy)
        self.zoom = min(max(self.zoom * factor, 1.0), 16.0)
        self.view_x0, self.view_y0 = ix - cx / self.zoom, iy - cy / self.zoom
        self._clamp_view()
        self._redraw = True

    def _on_mouse(self, event, x, y, flags, _param):
        # both panels share one view, so map either panel to view coordinates;
        # drawing only happens on the left (segmentation) panel
        w = self.image.shape[1]
        panel = 0 if x < w else 1
        if panel:
            x = max(0, x - w - SEPARATOR_PX)
        self.cursor = (x, y)
        ix, iy = self._to_image(x, y)
        if panel != 0:
            self._painting = False
            if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
                self._redraw = True
                return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == 'polygon':
                self.polygon.append((ix, iy))
            else:
                self._push_history()
                self._painting = True
                self._last_paint = (ix, iy)
                self._paint(self._last_paint, self._last_paint)
        elif event == cv2.EVENT_LBUTTONUP:
            self._painting = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._pan_from = (x, y, self.view_x0, self.view_y0)
        elif event == cv2.EVENT_RBUTTONUP:
            self._pan_from = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._painting:
                self._paint(self._last_paint, (ix, iy))
                self._last_paint = (ix, iy)
            if self._pan_from is not None:
                px, py, vx, vy = self._pan_from
                self.view_x0 = vx - (x - px) / self.zoom
                self.view_y0 = vy - (y - py) / self.zoom
                self._clamp_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            # wheel delta is the high 16 bits of flags (cv2.getMouseWheelDelta
            # isn't in every build), so its sign is the sign of flags
            self._zoom_at(x, y, 1.25 if flags > 0 else 0.8)
        self._redraw = True

    def _render(self):
        h, w = self.image.shape[:2]
        z = self.zoom
        composite = self.image
        if self.show_mask and self.mask.any():
            composite = self.image.copy()
            sel = self.mask > 0
            composite[sel] = (0.55 * self.image[sel] + 0.45 * numpy.array(MASK_COLOR)).astype(numpy.uint8)
        M = numpy.float32([[z, 0, -self.view_x0 * z], [0, z, -self.view_y0 * z]])
        interp = cv2.INTER_NEAREST if z >= 2 else cv2.INTER_LINEAR
        canvas = cv2.warpAffine(composite, M, (w, h), flags=interp)

        def to_canvas(points):
            return numpy.round((numpy.asarray(points, numpy.float64).reshape(-1, 2)
                                - [self.view_x0, self.view_y0]) * z).astype(numpy.int32)

        # right panel: same view, GT clevis silhouette rendered over the image
        gt_composite = self.image.copy()
        sel = self.gt_mask > 0
        gt_composite[sel] = (0.6 * self.image[sel] + 0.4 * numpy.array(GT_COLOR)).astype(numpy.uint8)
        gt_canvas = cv2.warpAffine(gt_composite, M, (w, h), flags=interp)
        contours, _ = cv2.findContours(self.gt_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            cv2.polylines(gt_canvas, [to_canvas(c)], True, GT_COLOR, 2, cv2.LINE_AA)
        label = self.reference_label
        if not sel.any():
            label += ' -- OFF-IMAGE'
        cv2.putText(gt_canvas, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(gt_canvas, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GT_COLOR, 2, cv2.LINE_AA)

        if self.show_mask:
            contours, _ = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                cv2.polylines(canvas, [to_canvas(c)], True, MASK_COLOR, 1, cv2.LINE_AA)
        if self.polygon:
            color = (255, 0, 255) if self.erase else (0, 255, 255)
            pts = to_canvas(self.polygon)
            cv2.polylines(canvas, [numpy.vstack([pts, [self.cursor]])], False, color, 1, cv2.LINE_AA)
            for p in pts:
                cv2.circle(canvas, tuple(int(c) for c in p), 3, color, -1)
        if self.mode == 'brush':
            cv2.circle(canvas, self.cursor, max(1, int(self.brush_radius * z)),
                       (255, 0, 255) if self.erase else (0, 255, 255), 1, cv2.LINE_AA)

        record = self.records[self.index]
        status = self._entry(record).get('status', 'pending').upper()
        lines = [
            INSTRUCTION,
            f'frame {record["frame_index"]} ({self.index + 1}/{len(self.records)}) [{status}]'
            f'{"  *unsaved*" if self.dirty else ""}   mode: {self.mode}{" (ERASE)" if self.erase else ""}'
            f'   brush: {self.brush_radius}px   zoom: {z:.1f}x',
            'click: add point  Enter/space: close polygon  Bksp: undo point  ESC: cancel polygon  b: brush  '
            'e: erase  [ ]: brush size',
            'wheel or =/-: zoom  right-drag: pan  0: reset view  h: hide mask  u: undo  c: clear  '
            's: save+next  n/p: next/prev  x: skip  q: quit',
        ]
        # the two-panel window is shown at about half scale, so larger text
        for i, line in enumerate(lines):
            y = 35 + i * 34
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        separator = numpy.full((h, SEPARATOR_PX, 3), 255, numpy.uint8)
        cv2.imshow(ANNOTATE_WINDOW_NAME, numpy.hstack([canvas, separator, gt_canvas]))

    def run(self):
        frames = self.annotations['frames']
        done = sum(1 for r in self.records if frames.get(str(r['frame_index']), {}).get('status') == 'saved')
        print(f'{len(self.records)} frame(s), {done} already segmented.')
        print(INSTRUCTION)
        print('Draw on the LEFT panel, tight to the visible metal; the right panel shows the GT clevis silhouette '
              'for comparison (same zoom/pan). Polygon (default): click points '
              'around the outline, Enter/space to fill it in (repeat for more pieces; e to switch to erase and '
              'cut pieces out). b: brush (drag to paint, [ ] size). Zoom with the wheel or =/-, pan with '
              'right-drag. s saves and moves on; n/p also save any changes. x skips a frame (clevis hidden), '
              'q quits (changes saved).')
        while True:
            if self._redraw:
                self._render()
                self._redraw = False
            key = cv2.waitKey(15)
            if key == -1:
                continue
            key &= 0xFF
            self._redraw = True
            if key == 27:  # ESC: cancel an in-progress polygon, else quit
                if self.polygon:
                    self.polygon = []
                    continue
                self._save_if_dirty()
                break
            elif key == ord('q'):
                self._save_if_dirty()
                break
            elif key in (13, 10, 32):
                self._close_polygon()
            elif key == 8:
                if self.polygon:
                    self.polygon.pop()
            elif key == ord('b'):
                self.mode = 'brush' if self.mode == 'polygon' else 'polygon'
                self.polygon = []
            elif key == ord('e'):
                self.erase = not self.erase
            elif key == ord('['):
                self.brush_radius = max(1, self.brush_radius - 1)
            elif key == ord(']'):
                self.brush_radius += 1
            elif key in (ord('='), ord('+')):
                self._zoom_at(*self.cursor, 1.25)
            elif key == ord('-'):
                self._zoom_at(*self.cursor, 0.8)
            elif key == ord('0'):
                self.zoom, self.view_x0, self.view_y0 = 1.0, 0.0, 0.0
            elif key == ord('h'):
                self.show_mask = not self.show_mask
            elif key == ord('u'):
                if self.history:
                    self.mask = self.history.pop()
                    self.dirty = True
            elif key == ord('c'):
                self._push_history()
                self.mask[:] = 0
            elif key == ord('s'):
                if not self.mask.any():
                    print('Mask is empty -- draw the clevis first, or x to skip this frame.')
                    continue
                self._save()
                self._go(1)
            elif key == ord('x'):
                self._save(status='skipped')
                self._go(1)
            elif key == ord('n'):
                self._go(1)
            elif key == ord('p'):
                self._go(-1)
        cv2.destroyWindow(ANNOTATE_WINDOW_NAME)


# ---------------------------------------------------------------------------
# phase 3: evaluate -> PDF
# ---------------------------------------------------------------------------

def crop_window(masks, size, margin_frac=0.6, min_margin_px=40):
    w, h = size
    ys, xs = numpy.nonzero(numpy.logical_or.reduce(masks))
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    margin = int(max(min_margin_px, margin_frac * max(x1 - x0, y1 - y0)))
    return (max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin))


def iou_panel(rgb, drawn, gt_mask):
    # overlap yellow, GT-only red, drawn-only green: exactly what the IoU counts
    out = rgb.astype(numpy.float32)
    for sel, color in ((drawn & gt_mask, (255, 220, 0)), (gt_mask & ~drawn, (255, 0, 0)),
                       (drawn & ~gt_mask, (0, 200, 0))):
        out[sel] = 0.45 * out[sel] + 0.55 * numpy.array(color)
    return out.astype(numpy.uint8)


def outline_panel(rgb, masks_colors):
    out = rgb.copy()
    for mask, color, thickness in masks_colors:
        contours, _ = cv2.findContours(mask.astype(numpy.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, color, thickness, cv2.LINE_AA)
    return out


def write_pdf(pdf_path, meta_lines, results, pages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    keys = [('iou', 'IoU'), ('rot_err_deg', 'rot err [deg]'), ('trans_err_mm', 'trans err [mm]'),
            ('trans_err_inplane_mm', 'in-plane [mm]'), ('trans_err_depth_mm', 'depth [mm]')]
    rows = [[str(r['frame_index'])] + [f'{r[k]:.3f}' if r.get(k) is not None else '-' for k, _ in keys]
            for r in results]
    for name, fn in (('mean', numpy.mean), ('median', numpy.median), ('std', numpy.std), ('max', numpy.max)):
        rows.append([name] + [f'{fn([r[k] for r in results if r.get(k) is not None]):.3f}'
                              if any(r.get(k) is not None for r in results) else '-' for k, _ in keys])

    with PdfPages(pdf_path) as pdf:
        per_page = 30
        for start in range(0, len(rows), per_page):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.96, 'Clevis ground truth vs hand segmentation', fontsize=15, weight='bold')
            fig.text(0.06, 0.94 - 0.016 * len(meta_lines), '\n'.join(meta_lines), fontsize=7.5, va='bottom',
                     family='monospace')
            ax = fig.add_axes([0.06, 0.06, 0.88, 0.80 - 0.016 * len(meta_lines)])
            ax.axis('off')
            chunk = rows[start:start + per_page]
            table = ax.table(cellText=chunk, colLabels=['frame'] + [label for _, label in keys], loc='upper center',
                             cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            table.scale(1, 1.3)
            for i, row in enumerate(chunk, start=1):
                if not row[0].isdigit():
                    for j in range(len(row)):
                        table[i, j].set_facecolor('#e8e8e8')
            pdf.savefig(fig)
            plt.close(fig)

        for r, (panel_iou, panel_err) in zip(results, pages):
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
            title = f'frame {r["frame_index"]}  ({r["image"]})   IoU {r["iou"]:.3f}'
            if r.get('rot_err_deg') is not None:
                dx, dy, dz = r['dxyz_mm']
                title += (f'\nrot err {r["rot_err_deg"]:.2f} deg   trans err {r["trans_err_mm"]:.2f} mm '
                          f'(x {dx:+.2f}, y {dy:+.2f}, z {dz:+.2f} mm; in-plane {r["trans_err_inplane_mm"]:.2f})')
            fig.suptitle(title, fontsize=10)
            axes[0].imshow(panel_iou)
            axes[0].set_title('IoU: GT clevis silhouette over the segmentation', fontsize=9)
            axes[0].legend(handles=[Patch(color=(1, 0.86, 0), label='overlap (GT and drawn)'),
                                    Patch(color=(1, 0, 0), label='GT only'),
                                    Patch(color=(0, 0.78, 0), label='drawn only')],
                           loc='lower right', fontsize=7)
            axes[1].imshow(panel_err)
            axes[1].set_title('Errors: GT clevis vs clevis pose best fitting the segmentation', fontsize=9)
            axes[1].legend(handles=[Patch(color=(1, 0, 0), label='GT silhouette'),
                                    Patch(color=(0, 1, 1), label='best-fit silhouette'),
                                    Patch(color=(0, 0.78, 0), label='drawn')],
                           loc='lower right', fontsize=7)
            for ax in axes:
                ax.axis('off')
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)


def run_evaluate(args, records, annotations) -> bool:
    frames = annotations['frames']
    to_eval = [r for r in records if frames.get(str(r['frame_index']), {}).get('status') == 'saved']
    if not to_eval:
        print('No segmented frames to evaluate.')
        return False

    if records[0].get('alignment_applied') is False:
        print('WARNING: this dataset was recorded with a hand-eye calibration that had NO 3D pose alignment '
              'applied -- the "GT" evaluated here is the raw hand-eye result.')

    model = ClevisModel(args.clevis_stl)
    rng = numpy.random.default_rng(args.seed)
    results, pages = [], []
    for n, record in enumerate(to_eval):
        idx = record['frame_index']
        started = time.time()
        drawn = cv2.imread(os.path.join(args.output_dir, frames[str(idx)]['mask']), cv2.IMREAD_GRAYSCALE) > 127
        image = cv2.imread(os.path.join(args.output_dir, record['right_rect_image']))
        K = numpy.array(record['camera_matrix_right_rect'], dtype=numpy.float64)
        size = (drawn.shape[1], drawn.shape[0])
        # the ground truth: post-alignment hand-eye @ kinematics, as recorded
        cam_T_clevis = cam_T_joint_model(numpy.array(record['cam_T_shaft_end'], dtype=numpy.float64),
                                         numpy.array(record['cam_T_gripper'], dtype=numpy.float64))

        gt_mask = render_silhouette(model, cam_T_clevis, K, size) > 127
        result = {'frame_index': idx, 'image': record['right_rect_image'], 'iou': iou(gt_mask, drawn)}
        fit_mask = None
        if gt_mask.any():
            window = crop_window([gt_mask, drawn], size, margin_frac=0.5)
            cam_T_clevis_fit = fit_pose_to_mask(model, K, drawn, cam_T_clevis, window, args.fit_restarts, rng)
            fit_mask = render_silhouette(model, cam_T_clevis_fit, K, size) > 127
            result.update(pose_error(cam_T_clevis, cam_T_clevis_fit))
            result['iou_fit'] = iou(fit_mask, drawn)
        else:
            print(f'  frame {idx}: GT clevis is off-image, no pose error')
        results.append(result)

        x0, y0, x1, y1 = crop_window([m for m in (gt_mask, drawn, fit_mask) if m is not None and m.any()], size)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
        c = (slice(y0, y1), slice(x0, x1))
        outlines = [(gt_mask[c], (255, 0, 0), 2), (drawn[c], (0, 200, 0), 1)]
        if fit_mask is not None:
            outlines.insert(1, (fit_mask[c], (0, 255, 255), 2))
        pages.append((iou_panel(rgb, drawn[c], gt_mask[c]), outline_panel(rgb, outlines)))

        msg = f'[{n + 1}/{len(to_eval)}] frame {idx}  IoU {result["iou"]:.3f}'
        if 'rot_err_deg' in result:
            msg += f'  rot {result["rot_err_deg"]:.2f}deg  trans {result["trans_err_mm"]:.2f}mm ' \
                   f'(in-plane {result["trans_err_inplane_mm"]:.2f}, depth {result["trans_err_depth_mm"]:.2f})'
        print(msg + f'  ({time.time() - started:.0f}s)')

    print(f'\nMean over {len(results)} frame(s): IoU {numpy.mean([r["iou"] for r in results]):.3f}', end='')
    with_err = [r for r in results if 'rot_err_deg' in r]
    if with_err:
        print(f', rot {numpy.mean([r["rot_err_deg"] for r in with_err]):.2f}deg, '
              f'trans {numpy.mean([r["trans_err_mm"] for r in with_err]):.2f}mm '
              f'(in-plane {numpy.mean([r["trans_err_inplane_mm"] for r in with_err]):.2f}, '
              f'depth {numpy.mean([r["trans_err_depth_mm"] for r in with_err]):.2f})')
    else:
        print()

    meta_lines = [
        f'dataset:       {os.path.abspath(args.output_dir)}',
        f'evaluated:     {time.ctime()}',
        f'ground truth:  {records[0].get("hand_eye_json", "hand-eye calibration")} base-frame '
        f'(alignment applied: {records[0].get("alignment_applied", "unknown")}) @ PSM kinematics',
        f'model:         {model.source} (clevis only)',
        'IoU:           GT clevis silhouette vs drawn mask',
        'errors:        GT clevis pose vs pose best fitting the drawn mask (camera frame, at the pitch pin);',
        '               depth = camera z, only weakly observable from a silhouette',
    ]
    pdf_path = os.path.join(args.output_dir, EVAL_PDF_NAME)
    write_pdf(pdf_path, meta_lines, results, pages)
    print(f'Saved {pdf_path}')
    return True


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Record a dataset, hand-segment the clevis, and score the hand-eye + alignment ground '
                    'truth against it (IoU, rotational and translational error) in a PDF.')
    parser.add_argument('-o', '--output-dir', type=str, default='gt_segmentation_dataset',
                        help='dataset directory (created if missing; re-running resumes into it). Any '
                             'image_generation_gt_hand_eye.py output directory works with --annotate-only.')
    parser.add_argument('--annotate-only', action='store_true',
                        help='skip recording: segment (then evaluate) an existing dataset, no ROS needed')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='skip recording and segmenting: just evaluate the existing masks')
    # record
    parser.add_argument('-p', '--psm-name', type=str, default='PSM1', choices=['PSM1', 'PSM2', 'PSM3'])
    parser.add_argument('-H', '--hand-eye-json', type=str, default=DEFAULT_HAND_EYE_JSON,
                        help=f'hand-eye calibration (with the GUI alignment applied) used as ground truth '
                             f'when recording (default: {DEFAULT_HAND_EYE_JSON})')
    parser.add_argument('-S', '--stereo-calib-dir', type=str, default=None,
                        help="left.yaml/right.yaml for stereo rectification "
                             "(default: image_generation_gt_hand_eye.py's DEFAULT_STEREO_CALIB_DIR)")
    parser.add_argument('--startup-delay', type=float, default=5.0)
    # evaluate
    parser.add_argument('--clevis-stl', type=str, default=DEFAULT_CLEVIS_STL)
    parser.add_argument('--fit-restarts', type=int, default=4,
                        help='pose-fit starts per frame: the GT pose plus N-1 perturbed ones (default: 4)')
    parser.add_argument('--seed', type=int, default=0)

    offline = any(flag in sys.argv[1:] for flag in ('--annotate-only', '--evaluate-only'))
    if offline:
        argv = sys.argv[1:]
    else:
        import crtk
        argv = crtk.ral.parse_argv(sys.argv[1:])  # also calls rclpy.init(), skip argv[0]
    args = parser.parse_args(argv)

    if not offline and not run_record(args):
        return

    records = load_gt_records(args.output_dir)
    annotations = load_annotations(args.output_dir)
    if not args.evaluate_only:
        save_annotations(args.output_dir, annotations)
        Annotator(args.output_dir, records, annotations, ClevisModel(args.clevis_stl)).run()

    if run_evaluate(args, records, annotations):
        cleanup(args.output_dir, records, annotations)


if __name__ == '__main__':
    main()
