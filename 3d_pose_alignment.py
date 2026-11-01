#!/usr/bin/env python3

# Offline GUI for refining a dVRK hand-eye calibration by visual 3D pose
# alignment, inspired by the "Adjust 3D model!" GUI from the SurgRIPE
# challenge paper (Xu et al., Medical Image Analysis 105 (2025) 103674,
# Fig. 6). That paper uses the GUI to manually tune a single FIXED
# marker-to-instrument offset until a 3D model projection overlays a
# physical keydot marker's tracked instrument in every frame.
#
# This script adapts the same interaction (OpenCV trackbars for a 6DoF
# Trans/Rot correction, scrubbing through frames) to a different problem:
# there is no marker here, since dVRK kinematics (measured_cp) already give
# the gripper's pose directly. What's uncertain is the hand-eye calibration
# (cam_T_base) itself, which was used by hand_eye_gt_raw.py to project a
# debug axis onto each captured frame at two reference points -- the
# gripper tip and the end of the shaft (before the wrist), matching
# vis_gripper_pose.py's own two-axis display. For each frame this GUI lets
# you nudge a single 6DoF correction (applied identically to both
# reference points, since both come from the same hand-eye calibration)
# until a translucent silhouette spanning the two points lands on the real
# tool in the image, and saves that correction. Unlike the paper's single
# shared offset, each frame gets its own independent correction here --
# each is one noisy sample of the true cam_T_base, collected at a
# different robot/camera configuration. On quit, all saved corrections
# (across every dataset passed on the command line) are combined into one
# refined cam_T_base -- translation by arithmetic mean, rotation by SVD
# orthonormalization of the mean rotation matrix -- and you're asked
# whether to write it back into the hand-eye calibration file.

import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy
from scipy.spatial.transform import Rotation

WINDOW_NAME = 'Adjust 3D model!'
SLIDER_MAX = 1000  # more steps than the visible pixel width buys finer control over a big range
SLIDER_CENTER = SLIDER_MAX // 2
RANGE_STEPS = 10  # 'Range x' trackbar goes from 1x to RANGE_STEPS x
DRAG_GRAB_RADIUS_PX = 20  # how close a click must land to a rendered point to grab it

GT_JSON_NAME = 'hand_eye_gt_raw.json'
CORRECTIONS_JSON_NAME = 'hand_eye_gt_raw_corrections.json'

# camera_registration.py always writes PSM1's hand-eye calibration here,
# and hand_eye_gt_raw.py defaults to reading it from the same place
DEFAULT_HAND_EYE_JSON = os.path.expanduser('~/PSM1-registration-open-cv.json')


def draw_axis(img, camera_matrix, dist_coeffs, pose, size=0.01):
    # Mirrors hand_eye_gt_raw.py's draw_axis (itself mirroring
    # dvrk_camera_registration's vis_gripper_pose.py), so the overlay here
    # matches the one already baked into each frame's *_overlay image.
    thickness = 2
    rotation, translation = pose[:3, :3], pose[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)
    points = numpy.float32([[size, 0, 0], [0, size, 0], [0, 0, size], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rvec, translation, camera_matrix, dist_coeffs)
    axis_points = axis_points.astype(int)

    origin = tuple(int(v) for v in axis_points[3].ravel())
    img = cv2.line(img, origin, tuple(axis_points[0].ravel()), (255, 0, 0), thickness)
    img = cv2.line(img, origin, tuple(axis_points[1].ravel()), (0, 255, 0), thickness)
    img = cv2.line(img, origin, tuple(axis_points[2].ravel()), (0, 0, 255), thickness)
    return img, origin


def _project(points_3d, camera_matrix, dist_coeffs) -> numpy.ndarray:
    # points_3d are already expressed in the camera frame, so project with
    # zero rvec/tvec rather than re-applying a pose.
    zero = numpy.zeros(3)
    points_2d, _ = cv2.projectPoints(numpy.asarray(points_3d, dtype=numpy.float32),
                                      zero, zero, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2).astype(numpy.int32)


def _fill_translucent(img, points_2d, color, alpha):
    overlay = img.copy()
    cv2.fillConvexPoly(overlay, points_2d, color)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.polylines(img, [points_2d], True, color, 2, cv2.LINE_AA)
    return img


def draw_gripper_silhouette(img, camera_matrix, dist_coeffs, cam_T_shaft, cam_T_gripper,
                             jaw_length, jaw_half_angle_deg, half_width,
                             color=(0, 255, 255), alpha=0.35):
    # A plain cylinder is rotationally symmetric about its own axis, so a
    # roll error is invisible in it -- there's nothing asymmetric to catch
    # the eye. This draws two open jaw blades from the gripper point (like
    # a needle driver's tip) plus a pivot ring at the shaft point instead,
    # so roll/pitch/yaw errors show up as the blades visibly not matching
    # the real jaws' shape/orientation, not just as points being in the
    # right place. Still purely parametric (no CAD/joint-angle data needed)
    # -- an approximate visual aid, not exact tool geometry.
    shaft_origin = cam_T_shaft[0:3, 3]
    gripper_origin = cam_T_gripper[0:3, 3]

    # dVRK tool-tip frame: Z points out along the jaws, X is the yaw pin the
    # jaws pivot on, so they open/close in the Y-Z plane. The shaft-to-gripper
    # line is the clevis, not the jaws -- the two differ by the yaw angle.
    forward = cam_T_gripper[0:3, 2]
    open_dir = cam_T_gripper[0:3, 1]
    side_dir = cam_T_gripper[0:3, 0]

    half_angle = numpy.radians(jaw_half_angle_deg)
    for sign in (1.0, -1.0):
        blade_dir = numpy.cos(half_angle) * forward + sign * numpy.sin(half_angle) * open_dir
        blade_tip = gripper_origin + blade_dir * jaw_length
        quad_3d = [
            gripper_origin + side_dir * half_width,
            gripper_origin - side_dir * half_width,
            blade_tip - side_dir * half_width * 0.3,
            blade_tip + side_dir * half_width * 0.3,
        ]
        img = _fill_translucent(img, _project(quad_3d, camera_matrix, dist_coeffs), color, alpha)

    # pivot ring around the pitch pin (shaft-end Y) at the shaft point
    ring_3d = [
        shaft_origin + half_width * (numpy.cos(a) * cam_T_shaft[0:3, 0] + numpy.sin(a) * cam_T_shaft[0:3, 2])
        for a in numpy.linspace(0, 2 * numpy.pi, 14, endpoint=False)
    ]
    cv2.polylines(img, [_project(ring_3d, camera_matrix, dist_coeffs)], True, color, 2, cv2.LINE_AA)

    # tool body, a simple line from the pivot to between the two jaws
    img = cv2.line(img, tuple(_project([shaft_origin], camera_matrix, dist_coeffs)[0]),
                    tuple(_project([gripper_origin], camera_matrix, dist_coeffs)[0]), color, 2, cv2.LINE_AA)
    return img


def cam_T_joint_model(cam_T_shaft, cam_T_gripper) -> numpy.ndarray:
    # SurgRIPE's joint.npy/joint.stl is the wrist clevis -- the link between
    # the pitch and yaw pins, rigid with neither the shaft nor the jaws. In
    # its own frame (mm) the pitch pin hole is at the origin along X, the
    # yaw pin hole is 9.1mm forward at (0, -9.1, 0) along Z, and the body
    # runs forward along -Y. On the dVRK side the shaft-end frame's origin
    # sits on the pitch pin, the gripper (tool tip) frame's origin sits on
    # the yaw pin (also 9.1mm apart), and gripper X is the yaw pin's own
    # axis -- so the two axes together pin down the clevis pose completely,
    # roll included. Gripper Z is NOT used: it points along the jaws, which
    # swing about the yaw pin relative to the clevis. The mesh is symmetric
    # to ~0.05mm under a 180deg turn about its own Y, so mapping its Z to
    # +gripper X rather than -gripper X is immaterial.
    shaft_origin = cam_T_shaft[0:3, 3]
    forward = cam_T_gripper[0:3, 3] - shaft_origin
    forward = forward / numpy.linalg.norm(forward)
    z_axis = cam_T_gripper[0:3, 0] - forward * numpy.dot(cam_T_gripper[0:3, 0], forward)
    z_axis = z_axis / numpy.linalg.norm(z_axis)
    y_axis = -forward
    x_axis = numpy.cross(y_axis, z_axis)
    matrix = numpy.eye(4)
    matrix[0:3, 0:3] = numpy.column_stack([x_axis, y_axis, z_axis])
    matrix[0:3, 3] = shaft_origin
    return matrix


def draw_joint_model(img, camera_matrix, dist_coeffs, cam_T_anchor, points_mm, color=(255, 0, 255), dot_px=3):
    # Real CAD surface points (e.g. SurgRIPE's joint.npy) projected as a
    # scatter, anchored at cam_T_anchor (from cam_T_joint_model,
    # tying the model to BOTH axes at once). A hole in the real tool has no
    # surface there in the mesh either, so it shows up as a genuine gap in
    # the scatter with no occlusion handling needed -- unlike the
    # parametric jaw shape, this is the tool's actual geometry.
    points_m = numpy.asarray(points_mm, dtype=numpy.float64) / 1000.0  # CAD model units are mm
    points_cam = points_m @ cam_T_anchor[0:3, 0:3].T + cam_T_anchor[0:3, 3]

    points_2d = _project(points_cam, camera_matrix, dist_coeffs)
    h, w = img.shape[:2]
    xs, ys = points_2d[:, 0], points_2d[:, 1]
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if len(xs) == 0:
        return img

    mask = numpy.zeros((h, w), dtype=numpy.uint8)
    mask[ys, xs] = 255
    if dot_px > 1:
        mask = cv2.dilate(mask, numpy.ones((dot_px, dot_px), numpy.uint8), iterations=1)

    img = img.copy()
    img[mask > 0] = color
    return img


def delta_transform(trans_m, rot_deg, pivot=None) -> numpy.ndarray:
    # Without a pivot, the rotation turns around the camera's own origin --
    # since the tool sits tens of cm away from the camera, even a few
    # degrees then swings the whole silhouette across the image (dominated
    # by an apparent translation) instead of spinning it in place, making
    # orientation impossible to isolate from position. Passing the tool's
    # own (nominal, uncorrected) position as `pivot` fixes that: the
    # returned transform still composes as a normal left-multiplied SE(3)
    # correction (so it applies identically to any camera-frame pose, e.g.
    # both the gripper and shaft-end points, and to cam_T_base itself), but
    # rotates about `pivot` and translates by `trans_m` on top of that.
    R_delta = Rotation.from_euler('xyz', rot_deg, degrees=True).as_matrix()
    matrix = numpy.eye(4)
    matrix[0:3, 0:3] = R_delta
    if pivot is None:
        matrix[0:3, 3] = trans_m
    else:
        pivot = numpy.asarray(pivot)
        matrix[0:3, 3] = (numpy.eye(3) - R_delta) @ pivot + numpy.asarray(trans_m)
    return matrix


def average_transforms(matrices):
    # matrices: list of 4x4 cam_T_base estimates, one per saved correction --
    # each is one noisy sample of the same unknown fixed calibration.
    # Translation lives in R^3, so the arithmetic mean is exact. Rotations
    # don't average linearly; averaging the 3x3 rotation matrices elementwise
    # and re-orthonormalizing via SVD (the "chordal L2 mean") is a standard,
    # simple way to get a proper rotation back out, and is accurate as long
    # as the samples are reasonably close together -- true here since they're
    # all corrections to nearly the same starting calibration.
    translations = numpy.array([m[0:3, 3] for m in matrices])
    rotations = numpy.array([m[0:3, 0:3] for m in matrices])

    t_avg = translations.mean(axis=0)
    U, _, Vt = numpy.linalg.svd(rotations.sum(axis=0))
    R_avg = U @ Vt
    if numpy.linalg.det(R_avg) < 0:  # keep a proper rotation, not a reflection
        U[:, -1] *= -1
        R_avg = U @ Vt

    refined = numpy.eye(4)
    refined[0:3, 0:3] = R_avg
    refined[0:3, 3] = t_avg

    trans_dev_mm = numpy.linalg.norm(translations - t_avg, axis=1) * 1000.0
    rot_dev_deg = numpy.array([
        Rotation.from_matrix(R.T @ R_avg).magnitude() * 180.0 / numpy.pi
        for R in rotations
    ])
    diagnostics = {
        'trans_std_mm': float(trans_dev_mm.std()),
        'trans_max_mm': float(trans_dev_mm.max()),
        'rot_std_deg': float(rot_dev_deg.std()),
        'rot_max_deg': float(rot_dev_deg.max()),
    }
    return refined, diagnostics


def _rotation_procrustes(target_sum_outer: numpy.ndarray) -> numpy.ndarray:
    U, _, Vt = numpy.linalg.svd(target_sum_outer)
    R = U @ Vt
    if numpy.linalg.det(R) < 0:  # keep a proper rotation, not a reflection
        U[:, -1] *= -1
        R = U @ Vt
    return R


def solve_axb_c(C_list, M_list, iterations=300):
    # Solve C_k ~= X @ M_k @ Y for fixed 4x4 rigid transforms X, Y, given K
    # known (C_k, M_k) sample pairs. This is what actually lets multiple
    # camera angles combine into one better calibration here: C_k is the
    # measured cam_T_base for angle k, and M_k = inv(ecm_local_measured_cp)
    # for that angle -- since the camera is a fixed mechanical mount on the
    # ECM (X = cam_T_ecm_tip) and the two arms' RCMs don't move relative to
    # each other on this SUJ-fixed hardware (Y = ecm_rcm_T_psm1_rcm), X and
    # Y are genuinely angle-independent constants, even though cam_T_base
    # itself changes every time the ECM is driven to a new angle.
    #
    # Needs at least 2 samples with sufficiently different M_k rotations to
    # be well-posed -- a single camera angle gives only one effective M,
    # which can't separate X from Y (same reason plain hand-eye calibration
    # needs the robot to visit multiple, rotationally diverse poses).
    R_C = [m[0:3, 0:3] for m in C_list]
    t_C = [m[0:3, 3] for m in C_list]
    R_M = [m[0:3, 0:3] for m in M_list]
    t_M = [m[0:3, 3] for m in M_list]
    K = len(C_list)

    R_Y = numpy.eye(3)
    R_X = numpy.eye(3)
    for _ in range(iterations):
        # fix R_Y, solve R_X via orthogonal Procrustes; then vice versa
        B = [R_M[k] @ R_Y for k in range(K)]
        R_X = _rotation_procrustes(sum(R_C[k] @ B[k].T for k in range(K)))
        A = [R_X @ R_M[k] for k in range(K)]
        R_Y = _rotation_procrustes(sum(A[k].T @ R_C[k] for k in range(K)))

    # translations are linear given R_X, R_Y fixed:
    # t_C_k = R_X @ R_M_k @ t_Y + R_X @ t_M_k + t_X  ->  solve [t_Y; t_X] jointly
    rows, rhs = [], []
    for k in range(K):
        rows.append(numpy.hstack([R_X @ R_M[k], numpy.eye(3)]))
        rhs.append(t_C[k] - R_X @ t_M[k])
    solution, *_ = numpy.linalg.lstsq(numpy.vstack(rows), numpy.hstack(rhs), rcond=None)
    t_Y, t_X = solution[0:3], solution[3:6]

    X = numpy.eye(4)
    X[0:3, 0:3], X[0:3, 3] = R_X, t_X
    Y = numpy.eye(4)
    Y[0:3, 0:3], Y[0:3, 3] = R_Y, t_Y
    return X, Y


def axb_c_residuals(X, Y, C_list, M_list):
    trans_mm, rot_deg = [], []
    for C, M in zip(C_list, M_list):
        predicted = X @ M @ Y
        diff = numpy.linalg.inv(C) @ predicted
        trans_mm.append(float(numpy.linalg.norm(diff[0:3, 3]) * 1000.0))
        rot_deg.append(float(Rotation.from_matrix(diff[0:3, 0:3]).magnitude() * 180.0 / numpy.pi))
    return trans_mm, rot_deg


def load_hand_eye_calibration(json_file: str) -> numpy.ndarray:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return numpy.array(data['base-frame']['transform']).reshape(4, 4)


def write_hand_eye_calibration(json_file: str, cam_T_base: numpy.ndarray) -> str:
    with open(json_file, 'r') as f:
        data = json.load(f)
    data['base-frame']['transform'] = cam_T_base.tolist()

    backup_path = f'{json_file}.bak-{int(time.time())}'
    shutil.copy2(json_file, backup_path)
    with open(json_file, 'w') as f:
        json.dump(data, f)
    return backup_path


def slider_to_value(pos: int, rng: float) -> float:
    return (pos - SLIDER_CENTER) / SLIDER_CENTER * rng


def value_to_slider(value: float, rng: float) -> int:
    if rng == 0:
        return SLIDER_CENTER
    pos = int(round(value / rng * SLIDER_CENTER)) + SLIDER_CENTER
    return max(0, min(SLIDER_MAX, pos))


class Dataset:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))
        self.warned_no_shaft = False

        gt_json_path = os.path.join(path, GT_JSON_NAME)
        if not os.path.exists(gt_json_path):
            raise FileNotFoundError(f'{gt_json_path} not found -- is {path} a hand_eye_gt_raw.py output directory?')
        with open(gt_json_path, 'r') as f:
            data = json.load(f)
        self.meta = data['meta']
        self.records = data['records']
        if not self.records:
            raise ValueError(f'{gt_json_path} has no recorded frames')
        if 'right_camera_info' not in self.meta:
            raise ValueError(f'{gt_json_path} has no right_camera_info in its meta -- cannot project poses')

        p_matrix = numpy.array(self.meta['right_camera_info']['P']).reshape((3, 4))
        self.camera_matrix = p_matrix[0:3, 0:3]
        self.dist_coeffs = numpy.zeros((5, 1))
        self.cam_T_base = numpy.array(self.meta['cam_T_base']).reshape((4, 4))

        self.corrections_path = os.path.join(path, CORRECTIONS_JSON_NAME)
        self.corrections = {}
        self.skipped = set()
        if os.path.exists(self.corrections_path):
            with open(self.corrections_path, 'r') as f:
                saved = json.load(f)
            self.corrections = saved.get('corrections', {})
            self.skipped = set(saved.get('skipped', []))

    def image_path(self, record) -> str:
        return os.path.join(self.path, record['right_rect_image'])

    def has_shaft(self, record) -> bool:
        has = 'cam_T_shaft_end' in record
        if not has and not self.warned_no_shaft:
            print(f'{self.name}: frames have no cam_T_shaft_end (collected with an older '
                  f'hand_eye_gt_raw.py) -- showing single-axis overlay only, no silhouette. '
                  f'Re-run hand_eye_gt_raw.py to get the two-point silhouette for this dataset.')
            self.warned_no_shaft = True
        return has

    def save_correction(self, frame_index: int, trans_m, rot_deg, delta_T, cam_T_gripper_corrected):
        self.skipped.discard(str(frame_index))
        self.corrections[str(frame_index)] = {
            'trans_m': list(trans_m),
            'rot_deg': list(rot_deg),
            'delta_T': delta_T.tolist(),
            'cam_T_gripper_corrected': cam_T_gripper_corrected.tolist(),
            'cam_T_base_estimate': (delta_T @ self.cam_T_base).tolist(),
            'timestamp': time.time(),
        }
        self._save_state()

    def mark_skipped(self, frame_index: int):
        # "Skipped" and "corrected" are mutually exclusive, and a skipped
        # frame is removed from `corrections` entirely, so it's guaranteed
        # to be excluded from any later combination step, not just ignored.
        self.corrections.pop(str(frame_index), None)
        self.skipped.add(str(frame_index))
        self._save_state()

    def unmark_skipped(self, frame_index: int):
        if str(frame_index) in self.skipped:
            self.skipped.discard(str(frame_index))
            self._save_state()

    def _save_state(self):
        with open(self.corrections_path, 'w') as f:
            json.dump({
                'source': GT_JSON_NAME,
                'corrections': self.corrections,
                'skipped': sorted(self.skipped, key=int),
            }, f, indent=2)


class AlignmentGUI:
    def __init__(self, datasets, trans_range_m: float, rot_range_deg: float,
                 marker_size: float, jaw_length_m: float, jaw_half_angle_deg: float,
                 jaw_half_width_m: float, hand_eye_json: str, joint_points_mm=None, joint_dot_px: int = 3):
        self.datasets = datasets
        self.joint_points_mm = joint_points_mm
        self.joint_dot_px = joint_dot_px
        self.base_trans_range_m = trans_range_m
        self.base_rot_range_deg = rot_range_deg
        self.marker_size = marker_size
        self.jaw_length_m = jaw_length_m
        self.jaw_half_angle_deg = jaw_half_angle_deg
        self.jaw_half_width_m = jaw_half_width_m
        self.hand_eye_json = hand_eye_json

        # flattened (dataset, frame_index_in_dataset, record) across all datasets
        self.frames = [
            (ds, i, record)
            for ds in datasets
            for i, record in enumerate(ds.records)
        ]

        # authoritative current correction for the displayed frame; slider
        # positions are derived from this (see _set_trackbars_from_state),
        # not the other way around, so widening the range never jumps the
        # current pose.
        self.current_trans_m = [0.0, 0.0, 0.0]
        self.current_rot_deg = [0.0, 0.0, 0.0]
        self.range_multiplier = 1

        # drag state: which rendered point (if any) is currently grabbed,
        # and where the mouse was last, to turn drag deltas into pixel deltas
        self._drag_point = None
        self._drag_last_xy = None
        self._last_gripper_origin = None
        self._last_gripper_depth = None
        self._last_shaft_origin = None
        self._last_shaft_depth = None
        self._base_img = None

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Image', WINDOW_NAME, 0, len(self.frames) - 1, self._on_image_trackbar)
        cv2.createTrackbar('Range x', WINDOW_NAME, 0, RANGE_STEPS - 1, self._on_range_trackbar)
        cv2.createTrackbar('Trans X', WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Trans Y', WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Trans Z', WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Rot X', WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Rot Y', WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Rot Z', WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        self.current_index = -1
        self._load_frame(0)

    def _effective_ranges(self):
        return (self.base_trans_range_m * self.range_multiplier,
                self.base_rot_range_deg * self.range_multiplier)

    def _on_image_trackbar(self, pos):
        self._load_frame(pos)

    def _on_range_trackbar(self, pos):
        self.range_multiplier = pos + 1
        self._set_trackbars_from_state()

    def _on_pose_trackbar(self, _pos):
        trans_range_m, rot_range_deg = self._effective_ranges()
        self.current_trans_m = [
            slider_to_value(cv2.getTrackbarPos('Trans X', WINDOW_NAME), trans_range_m),
            slider_to_value(cv2.getTrackbarPos('Trans Y', WINDOW_NAME), trans_range_m),
            slider_to_value(cv2.getTrackbarPos('Trans Z', WINDOW_NAME), trans_range_m),
        ]
        self.current_rot_deg = [
            slider_to_value(cv2.getTrackbarPos('Rot X', WINDOW_NAME), rot_range_deg),
            slider_to_value(cv2.getTrackbarPos('Rot Y', WINDOW_NAME), rot_range_deg),
            slider_to_value(cv2.getTrackbarPos('Rot Z', WINDOW_NAME), rot_range_deg),
        ]
        self._render()

    def _current(self):
        return self.frames[self.current_index]

    def _load_frame(self, index):
        self.current_index = index
        ds, frame_index, record = self._current()
        saved = ds.corrections.get(str(frame_index))
        self.current_trans_m = list(saved['trans_m']) if saved else [0.0, 0.0, 0.0]
        self.current_rot_deg = list(saved['rot_deg']) if saved else [0.0, 0.0, 0.0]

        self._base_img = cv2.imread(ds.image_path(record))  # cached so drags/sliders don't re-hit disk

        self._widen_range_if_needed()  # so a saved correction (if any) is reachable by the sliders
        self._set_trackbars_from_state()

    def _widen_range_if_needed(self):
        needed = 1
        for value, base_range in ((v, self.base_trans_range_m) for v in self.current_trans_m):
            if base_range > 0:
                needed = max(needed, int(numpy.ceil(abs(value) / base_range)))
        for value, base_range in ((v, self.base_rot_range_deg) for v in self.current_rot_deg):
            if base_range > 0:
                needed = max(needed, int(numpy.ceil(abs(value) / base_range)))
        needed = min(needed, RANGE_STEPS)
        self.range_multiplier = needed
        cv2.setTrackbarPos('Range x', WINDOW_NAME, needed - 1)  # triggers _on_range_trackbar

    def _set_trackbars_from_state(self):
        trans_range_m, rot_range_deg = self._effective_ranges()
        cv2.setTrackbarPos('Trans X', WINDOW_NAME, value_to_slider(self.current_trans_m[0], trans_range_m))
        cv2.setTrackbarPos('Trans Y', WINDOW_NAME, value_to_slider(self.current_trans_m[1], trans_range_m))
        cv2.setTrackbarPos('Trans Z', WINDOW_NAME, value_to_slider(self.current_trans_m[2], trans_range_m))
        cv2.setTrackbarPos('Rot X', WINDOW_NAME, value_to_slider(self.current_rot_deg[0], rot_range_deg))
        cv2.setTrackbarPos('Rot Y', WINDOW_NAME, value_to_slider(self.current_rot_deg[1], rot_range_deg))
        cv2.setTrackbarPos('Rot Z', WINDOW_NAME, value_to_slider(self.current_rot_deg[2], rot_range_deg))
        self._render()

    def _sync_trans_sliders(self):
        # like _set_trackbars_from_state, but translation only -- used after
        # a drag, which never touches rotation
        trans_range_m, _ = self._effective_ranges()
        cv2.setTrackbarPos('Trans X', WINDOW_NAME, value_to_slider(self.current_trans_m[0], trans_range_m))
        cv2.setTrackbarPos('Trans Y', WINDOW_NAME, value_to_slider(self.current_trans_m[1], trans_range_m))
        cv2.setTrackbarPos('Trans Z', WINDOW_NAME, value_to_slider(self.current_trans_m[2], trans_range_m))

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            candidates = [('gripper', self._last_gripper_origin), ('shaft', self._last_shaft_origin)]
            best_point, best_dist = None, DRAG_GRAB_RADIUS_PX
            for name, origin in candidates:
                if origin is None:
                    continue
                dist = ((origin[0] - x) ** 2 + (origin[1] - y) ** 2) ** 0.5
                if dist < best_dist:
                    best_point, best_dist = name, dist
            if best_point is not None:
                self._drag_point = best_point
                self._drag_last_xy = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drag_point is not None:
                dx_px = x - self._drag_last_xy[0]
                dy_px = y - self._drag_last_xy[1]
                self._drag_last_xy = (x, y)
                if dx_px != 0 or dy_px != 0:
                    self._apply_drag(dx_px, dy_px)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag_point = None
            self._drag_last_xy = None

    def _apply_drag(self, dx_px, dy_px):
        # Convert a 2D pixel drag into a camera-frame X/Y translation at the
        # grabbed point's current depth (pinhole model: du/dX = fx/Z), and
        # leave Z untouched -- dragging on a 2D image has no natural mapping
        # to depth, same reason AR/CAD tools separate lateral drag from a
        # depth control. Rotation is never touched by a drag.
        ds, _frame_index, _record = self._current()
        depth = self._last_gripper_depth if self._drag_point == 'gripper' else self._last_shaft_depth
        if depth is None:
            return
        fx, fy = ds.camera_matrix[0, 0], ds.camera_matrix[1, 1]
        self.current_trans_m[0] += dx_px * depth / fx
        self.current_trans_m[1] += dy_px * depth / fy
        self._widen_range_if_needed()
        self._sync_trans_sliders()
        self._render()

    def _render(self):
        ds, frame_index, record = self._current()
        if self._base_img is None:
            print('Could not read the current frame\'s image.')
            return
        img = self._base_img.copy()

        cam_T_gripper_nominal = numpy.array(record['cam_T_gripper']).reshape((4, 4))
        pivot = cam_T_gripper_nominal[0:3, 3]  # rotate about the tool's own (nominal) position
        delta_T = delta_transform(self.current_trans_m, self.current_rot_deg, pivot=pivot)
        cam_T_gripper_corrected = delta_T @ cam_T_gripper_nominal

        if ds.has_shaft(record):
            cam_T_shaft_nominal = numpy.array(record['cam_T_shaft_end']).reshape((4, 4))
            cam_T_shaft_corrected = delta_T @ cam_T_shaft_nominal
            if self.joint_points_mm is not None:
                cam_T_joint_anchor = cam_T_joint_model(cam_T_shaft_corrected, cam_T_gripper_corrected)
                img = draw_joint_model(img, ds.camera_matrix, ds.dist_coeffs,
                                        cam_T_joint_anchor, self.joint_points_mm, dot_px=self.joint_dot_px)
            else:
                img = draw_gripper_silhouette(img, ds.camera_matrix, ds.dist_coeffs,
                                               cam_T_shaft_corrected, cam_T_gripper_corrected,
                                               self.jaw_length_m, self.jaw_half_angle_deg, self.jaw_half_width_m)
            img, shaft_origin = draw_axis(img, ds.camera_matrix, ds.dist_coeffs,
                                           cam_T_shaft_corrected, self.marker_size)
            img = cv2.circle(img, shaft_origin, 5, (255, 255, 0), -1)
            self._last_shaft_origin = shaft_origin
            self._last_shaft_depth = float(cam_T_shaft_corrected[2, 3])
        else:
            self._last_shaft_origin = None
            self._last_shaft_depth = None

        img, gripper_origin = draw_axis(img, ds.camera_matrix, ds.dist_coeffs,
                                         cam_T_gripper_corrected, self.marker_size)
        img = cv2.circle(img, gripper_origin, 6, (0, 0, 255), -1)
        self._last_gripper_origin = gripper_origin
        self._last_gripper_depth = float(cam_T_gripper_corrected[2, 3])

        if str(frame_index) in ds.skipped:
            status = 'SKIPPED'
        elif str(frame_index) in ds.corrections:
            status = 'SAVED'
        else:
            status = 'unsaved'
        lines = [
            f'{ds.name}  frame {frame_index} ({self.current_index + 1}/{len(self.frames)} total)  [{status}]',
            f'range: +/-{self.base_trans_range_m*1000*self.range_multiplier:.0f}mm  '
            f'+/-{self.base_rot_range_deg*self.range_multiplier:.0f}deg  ({self.range_multiplier}x)',
            f'trans (mm): x={self.current_trans_m[0]*1000:+.2f} y={self.current_trans_m[1]*1000:+.2f} '
            f'z={self.current_trans_m[2]*1000:+.2f}',
            f'rot (deg): x={self.current_rot_deg[0]:+.2f} y={self.current_rot_deg[1]:+.2f} '
            f'z={self.current_rot_deg[2]:+.2f}',
            'drag a dot to move it  [s]ave  [r]eset  [x]skip  [n]ext  [p]rev  [q]uit',
        ]
        for i, line in enumerate(lines):
            y = 25 + i * 25
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        self._last_delta_T = delta_T
        self._last_cam_T_gripper_corrected = cam_T_gripper_corrected
        cv2.imshow(WINDOW_NAME, img)

    def _save_current(self):
        ds, frame_index, _record = self._current()
        ds.save_correction(frame_index, self.current_trans_m, self.current_rot_deg,
                            self._last_delta_T, self._last_cam_T_gripper_corrected)
        print(f'Saved correction for {ds.name} frame {frame_index}: '
              f'trans(mm)={[round(v*1000, 2) for v in self.current_trans_m]} '
              f'rot(deg)={[round(v, 2) for v in self.current_rot_deg]}')
        self._render()

    def _reset_current(self):
        ds, frame_index, _record = self._current()
        ds.unmark_skipped(frame_index)
        self.current_trans_m = [0.0, 0.0, 0.0]
        self.current_rot_deg = [0.0, 0.0, 0.0]
        self.range_multiplier = 1
        cv2.setTrackbarPos('Range x', WINDOW_NAME, 0)  # triggers _on_range_trackbar -> _set_trackbars_from_state
        self._set_trackbars_from_state()

    def _skip_current(self):
        ds, frame_index, _record = self._current()
        ds.mark_skipped(frame_index)
        print(f'Skipped {ds.name} frame {frame_index} -- excluded from any hand-eye update.')
        self._go(1)  # move on, since a skipped frame is done with

    def _go(self, delta):
        new_index = max(0, min(len(self.frames) - 1, self.current_index + delta))
        cv2.setTrackbarPos('Image', WINDOW_NAME, new_index)  # triggers _load_frame

    def run(self):
        total_saved = sum(len(ds.corrections) for ds in self.datasets)
        total_skipped = sum(len(ds.skipped) for ds in self.datasets)
        print(f'{len(self.frames)} frames loaded from {len(self.datasets)} dataset(s), '
              f'{total_saved} corrections already saved, {total_skipped} skipped.')
        print('Drag the yellow (shaft) or red (gripper) dot to move it, or use the Trans/Rot sliders '
              '(dragging only moves things left/right/up/down, not rotation or depth), then press s to save.')
        print("Widen 'Range x' if the tool is too far off to reach at 1x.")
        print("Press x to skip a frame not worth correcting -- it's excluded from any hand-eye update.")
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('s'):
                self._save_current()
            elif key == ord('r'):
                self._reset_current()
            elif key == ord('x'):
                self._skip_current()
            elif key == ord('n'):
                self._go(1)
            elif key == ord('p'):
                self._go(-1)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()
        self._maybe_update_hand_eye()

    def _maybe_update_hand_eye(self):
        # Average within each dataset first -- cam_T_base itself changes
        # every time the ECM is driven to a new angle, so per-frame
        # corrections are only comparable within the SAME dataset. Also
        # pull each dataset's own (averaged) ECM pose: that's the piece
        # that changes with camera angle and lets multiple datasets be
        # combined properly below (see solve_axb_c).
        per_dataset = []
        for ds in self.datasets:
            frame_indices = [int(k) for k in ds.corrections.keys()]
            if not frame_indices:
                continue
            C_matrices = [numpy.array(ds.corrections[str(i)]['cam_T_base_estimate']).reshape((4, 4))
                          for i in frame_indices]
            ecm_matrices = [numpy.array(ds.records[i]['poses']['ecm_local_measured_cp']['matrix']).reshape((4, 4))
                            for i in frame_indices]
            C_avg, C_diag = average_transforms(C_matrices)
            ecm_avg, _ = average_transforms(ecm_matrices)
            per_dataset.append({
                'ds': ds, 'C': C_avg, 'M': numpy.linalg.inv(ecm_avg), 'diag': C_diag, 'n': len(frame_indices),
            })

        if not per_dataset:
            print('No corrections saved -- nothing to update in the hand-eye calibration.')
            return
        if not os.path.exists(self.hand_eye_json):
            print(f'{self.hand_eye_json} does not exist -- cannot update it. '
                  f'Pass --hand-eye-json to point at the right file.')
            return

        print()
        for entry in per_dataset:
            diag = entry['diag']
            print(f"{entry['ds'].name}: {entry['n']} correction(s), spread std {diag['trans_std_mm']:.2f}mm / "
                  f"{diag['rot_std_deg']:.2f}deg (max {diag['trans_max_mm']:.2f}mm / {diag['rot_max_deg']:.2f}deg)")

        if len(per_dataset) >= 2:
            # Multiple camera angles: cam_T_base differs at each one, but
            # the camera's fixed mount on the ECM and the (SUJ-fixed, so
            # also fixed) relationship between the two arms' RCMs don't --
            # jointly solving for those two lets every dataset's data
            # sharpen the same answer instead of only trusting one angle.
            C_list = [entry['C'] for entry in per_dataset]
            M_list = [entry['M'] for entry in per_dataset]
            X, Y = solve_axb_c(C_list, M_list)
            trans_res_mm, rot_res_deg = axb_c_residuals(X, Y, C_list, M_list)
            print(f'\nSolved a joint calibration across all {len(per_dataset)} datasets '
                  f'(camera-to-scope mount + scope-to-PSM1 mount, both angle-independent):')
            for entry, tr, ro in zip(per_dataset, trans_res_mm, rot_res_deg):
                print(f"  {entry['ds'].name}: solve vs. this dataset's own measurement differs by "
                      f'{tr:.2f}mm / {ro:.2f}deg')
            if max(rot_res_deg) > 10.0 or max(trans_res_mm) > 15.0:
                print('  These residuals are large -- the datasets don\'t self-consistently agree on one '
                      'answer yet. Treat the result below with caution; redoing the alignment pass (rotation '
                      'control was buggy before) or adding more angles should tighten this up.')

            names = [entry['ds'].name for entry in per_dataset]
            choice = input(f"\nWhich dataset matches your CURRENT camera angle ({', '.join(names)})? "
                            f"The joint solve will be applied to that angle's own ECM pose, or leave blank to "
                            f"skip: ").strip()
            chosen = next((entry for entry in per_dataset if entry['ds'].name == choice), None)
            if chosen is None:
                print('Not updating the hand-eye calibration file.')
                return
            refined = X @ chosen['M'] @ Y
        else:
            refined = per_dataset[0]['C']
            print(f"\nOnly one dataset has corrections -- using {per_dataset[0]['ds'].name}'s own average "
                  f'(need at least 2 datasets at different angles to jointly solve for an angle-independent '
                  f'calibration).')

        print(f'Hand-eye calibration file: {self.hand_eye_json}')
        answer = input('Update this hand-eye calibration file with the result above? [y/N]: ').strip().lower()
        if answer != 'y':
            print('Not updating the hand-eye calibration file.')
            return

        backup_path = write_hand_eye_calibration(self.hand_eye_json, refined)
        print(f'Updated {self.hand_eye_json} (previous version backed up to {backup_path}).')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'datasets', type=str, nargs='+',
        help='one or more hand_eye_gt_raw.py output directories (one per camera angle)',
    )
    parser.add_argument(
        '--trans-range-mm', type=float, default=50.0,
        help='translation slider range in mm at 1x, i.e. each slider spans +/- this value (default: 50mm)',
    )
    parser.add_argument(
        '--rot-range-deg', type=float, default=180.0,
        help='rotation slider range in degrees at 1x, i.e. each slider spans +/- this value '
             '(default: 180deg, i.e. full range without needing Range x)',
    )
    parser.add_argument(
        '--marker-size', type=float, default=0.01,
        help='axis marker length in meters, matches hand_eye_gt_raw.py draw_axis default (default: 0.01)',
    )
    parser.add_argument(
        '--jaw-length-mm', type=float, default=10.0,
        help='length of the drawn jaw blades from the gripper point, in mm '
             '(default: 10mm, roughly a needle-driver jaw)',
    )
    parser.add_argument(
        '--jaw-angle-deg', type=float, default=12.0,
        help='half-angle between the two drawn jaw blades, in degrees (default: 12deg)',
    )
    parser.add_argument(
        '--jaw-width-mm', type=float, default=4.0,
        help='width of the drawn jaw blades and the shaft-point pivot ring, in mm (default: 4mm)',
    )
    parser.add_argument(
        '-H', '--hand-eye-json', type=str, default=DEFAULT_HAND_EYE_JSON,
        help='hand-eye calibration file to offer to update with the combined corrections on quit '
             f'(default: {DEFAULT_HAND_EYE_JSON})',
    )
    parser.add_argument(
        '--joint-model', type=str, default=None,
        help='path to a .npy Nx3 point cloud of the tool joint (e.g. SurgRIPE\'s joint.npy, in mm), '
             'placed on the wrist clevis from the shaft-end and gripper axes together (see cam_T_joint_model), '
             'drawn instead of the parametric jaw shape if given',
    )
    parser.add_argument(
        '--joint-model-dot-px', type=int, default=3,
        help='pixel size of each drawn joint-model point after dilation (default: 3, reads as a solid '
             'shape so orientation is easy to see; 1-2 shows more true surface detail/holes but is fainter)',
    )
    args = parser.parse_args(sys.argv[1:])

    datasets = [Dataset(path) for path in args.datasets]

    joint_points_mm = numpy.load(args.joint_model) if args.joint_model else None

    gui = AlignmentGUI(
        datasets, args.trans_range_mm / 1000.0, args.rot_range_deg, args.marker_size,
        args.jaw_length_mm / 1000.0, args.jaw_angle_deg, args.jaw_width_mm / 1000.0 / 2.0,
        args.hand_eye_json, joint_points_mm, args.joint_model_dot_px,
    )
    gui.run()


if __name__ == '__main__':
    main()
