#!/usr/bin/env python3

# Captures a small dataset (default 20 frames) of synced left/right/rectified
# images together with PSM1/ECM/SUJ raw poses -- same rostopics and PSM1 crtk
# calls as hand_eye_gt_raw.py -- then walks through the captured frames in a
# manual alignment pass to also record a manually-aligned 6D pose in the
# camera frame for each one.
#
# Two phases, run back-to-back in one sitting by default:
#
#   1. Capture: a live preview window (rectified right image) waits for a
#      keypress. Press 'c' to snapshot a frame (images + raw poses saved
#      immediately, like hand_eye_gt_raw.py's automatic capture, but
#      keypress-triggered instead of timer-based). Press 'q' once you have
#      enough (or the target frame count is reached) to move on to
#      alignment, or ESC to abort without aligning (nothing captured is
#      lost either way -- each frame is saved to disk as soon as it's
#      captured).
#
#   2. Align: the same trackbar/drag interaction as 3d_pose_alignment.py's
#      "Adjust 3D model!" GUI, scrubbing through just-captured frames. For
#      each one, nudge the existing hand-eye calibration's projected
#      gripper/shaft-end axes until they visually land on the real tool,
#      then 's' to save, 'r' to reset, 'x' to skip, 'n'/'p' to move between
#      frames, 'q' to quit (progress is saved after every s/r/x, so quitting
#      partway through is safe).
#
# Pass --align-only to skip phase 1 and re-run phase 2 later on an existing
# output directory -- this needs no ROS/dVRK connection at all, since the
# camera intrinsics and raw poses were already saved to the dataset's JSON
# file, exactly like 3d_pose_alignment.py's own fully offline design.
#
# The manually-aligned poses are recorded as Rodrigues vectors
# (tx, ty, tz in meters, rx, ry, rz as an OpenCV/cv2.Rodrigues axis-angle
# vector in radians) in addition to the full 4x4 matrix, alongside the raw
# kinematic poses (kept in the same matrix/quaternion form as
# hand_eye_gt_raw.py) and the intrinsics/hand-eye calibration used.
#
# After the alignment pass, you're offered the chance to apply the saved
# corrections back into the hand-eye calibration file itself -- same
# averaging logic as 3d_pose_alignment.py's _maybe_update_hand_eye (mean
# translation, SVD-orthonormalized mean rotation across the saved frames),
# just folded into this script and scoped to a single dataset/camera angle
# rather than that script's multi-dataset joint solve.

import argparse
import json
import os
import shutil
import sys
import time

import cv2
import cv_bridge
import geometry_msgs.msg
import message_filters
import numpy
import sensor_msgs.msg
from scipy.spatial.transform import Rotation

import crtk
import dvrk_camera_registration

RIG = os.environ.get('RIG', 'jhu_dVRK')  # falls back if $RIG isn't set

# camera_registration.py always writes PSM1's hand-eye calibration here
DEFAULT_HAND_EYE_JSON = os.path.expanduser('~/PSM1-registration-open-cv.json')

# name -> topic, for the raw PoseStamped topics (besides PSM1's, which come
# from the crtk ARM handle below since we also need its forward_kinematics
# service) to record alongside each frame -- same set as hand_eye_gt_raw.py
POSE_TOPICS = {
    'ecm_measured_cp': '/ECM/measured_cp',
    'ecm_local_measured_cp': '/ECM/local/measured_cp',
    'suj_psm1_measured_cp': '/SUJ/PSM1/measured_cp',
    'suj_psm1_local_measured_cp': '/SUJ/PSM1/local/measured_cp',
    'suj_ecm_measured_cp': '/SUJ/ECM/measured_cp',
    'suj_ecm_local_measured_cp': '/SUJ/ECM/local/measured_cp',
}

GRIPPER_POSE_NAME = 'psm1_local_measured_cp'
SHAFT_POSE_NAME = 'psm1_shaft_end_local_cp'

DATASET_JSON_NAME = 'hand_eye_rodrigues_dataset.json'
IMAGE_SUBDIRS = ('left', 'right', 'right_rect', 'overlay_nominal', 'overlay_manual')

LIVE_WINDOW_NAME = 'Live capture (c: capture, q: finish & align, ESC: abort)'
ALIGN_WINDOW_NAME = 'Adjust manual pose (s: save, r: reset, x: skip, n/p: next/prev, q: quit)'
SLIDER_MAX = 1000
SLIDER_CENTER = SLIDER_MAX // 2
RANGE_STEPS = 10
DRAG_GRAB_RADIUS_PX = 20


# ---------------------------------------------------------------------------
# pose / intrinsics helpers (shared logic with hand_eye_gt_raw.py)
# ---------------------------------------------------------------------------

def load_hand_eye_calibration(json_file: str) -> numpy.ndarray:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return numpy.array(data['base-frame']['transform']).reshape(4, 4)


def quaternion_position_to_matrix(quat_xyzw, position) -> numpy.ndarray:
    matrix = numpy.eye(4)
    matrix[0:3, 0:3] = numpy.float64(Rotation.from_quat(quat_xyzw).as_matrix())
    matrix[0:3, 3] = position
    return matrix


def pose_stamped_to_matrix(msg: geometry_msgs.msg.PoseStamped) -> numpy.ndarray:
    q = msg.pose.orientation
    p = msg.pose.position
    return quaternion_position_to_matrix((q.x, q.y, q.z, q.w), (p.x, p.y, p.z))


def kdl_frame_to_matrix(frame) -> numpy.ndarray:
    return quaternion_position_to_matrix(frame.M.GetQuaternion(), (frame.p[0], frame.p[1], frame.p[2]))


def pose_record(msg: geometry_msgs.msg.PoseStamped, recv_time: float, now: float) -> dict:
    q = msg.pose.orientation
    p = msg.pose.position
    return {
        'matrix': pose_stamped_to_matrix(msg).tolist(),
        'position': [p.x, p.y, p.z],
        'orientation_xyzw': [q.x, q.y, q.z, q.w],
        'frame_id': msg.header.frame_id,
        'stamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        'age': now - recv_time,
    }


def kdl_pose_record(frame, now: float) -> dict:
    matrix = kdl_frame_to_matrix(frame)
    return {
        'matrix': matrix.tolist(),
        'position': matrix[0:3, 3].tolist(),
        'orientation_xyzw': list(frame.M.GetQuaternion()),
        'frame_id': '',
        'stamp': now,
        'age': 0.0,
    }


def camera_info_record(msg: sensor_msgs.msg.CameraInfo) -> dict:
    return {
        'width': msg.width,
        'height': msg.height,
        'distortion_model': msg.distortion_model,
        'D': list(msg.d),
        'K': list(msg.k),
        'R': list(msg.r),
        'P': list(msg.p),
        'frame_id': msg.header.frame_id,
    }


def matrix_to_rodrigues(matrix: numpy.ndarray) -> dict:
    # tx,ty,tz in meters; rx,ry,rz is a cv2.Rodrigues axis-angle vector in
    # radians (direction = rotation axis, magnitude = rotation angle) --
    # NOT independent per-axis angles.
    rvec, _ = cv2.Rodrigues(numpy.asarray(matrix[0:3, 0:3]))
    rvec = rvec.ravel()
    tvec = matrix[0:3, 3]
    return {
        'tx': float(tvec[0]), 'ty': float(tvec[1]), 'tz': float(tvec[2]),
        'rx': float(rvec[0]), 'ry': float(rvec[1]), 'rz': float(rvec[2]),
    }


# ---------------------------------------------------------------------------
# rendering helpers (mirrors hand_eye_gt_raw.py / 3d_pose_alignment.py so the
# overlay here matches what those scripts already draw)
# ---------------------------------------------------------------------------

def draw_axis(img, camera_matrix, dist_coeffs, pose, size=0.01):
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
    # Same parametric jaw-blade visual aid as 3d_pose_alignment.py's default
    # rendering: two open blades from the gripper point plus a pivot ring at
    # the shaft point, so roll/pitch/yaw errors show up as a visible shape
    # mismatch, not just points being in the right place.
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

    ring_3d = [
        shaft_origin + half_width * (numpy.cos(a) * cam_T_shaft[0:3, 0] + numpy.sin(a) * cam_T_shaft[0:3, 2])
        for a in numpy.linspace(0, 2 * numpy.pi, 14, endpoint=False)
    ]
    cv2.polylines(img, [_project(ring_3d, camera_matrix, dist_coeffs)], True, color, 2, cv2.LINE_AA)

    img = cv2.line(img, tuple(_project([shaft_origin], camera_matrix, dist_coeffs)[0]),
                    tuple(_project([gripper_origin], camera_matrix, dist_coeffs)[0]), color, 2, cv2.LINE_AA)
    return img


def draw_full_overlay(img, camera_matrix, dist_coeffs, cam_T_shaft, cam_T_gripper,
                       marker_size, jaw_length_m, jaw_half_angle_deg, jaw_half_width_m):
    img = draw_gripper_silhouette(img, camera_matrix, dist_coeffs, cam_T_shaft, cam_T_gripper,
                                   jaw_length_m, jaw_half_angle_deg, jaw_half_width_m)
    img, shaft_origin = draw_axis(img, camera_matrix, dist_coeffs, cam_T_shaft, marker_size)
    img = cv2.circle(img, shaft_origin, 5, (255, 255, 0), -1)
    img, gripper_origin = draw_axis(img, camera_matrix, dist_coeffs, cam_T_gripper, marker_size)
    img = cv2.circle(img, gripper_origin, 6, (0, 0, 255), -1)
    return img, shaft_origin, gripper_origin


def delta_transform(trans_m, rot_deg, pivot=None) -> numpy.ndarray:
    # See 3d_pose_alignment.py's delta_transform for why a pivot matters:
    # without one, small rotations dominated by apparent translation are
    # impossible to isolate from position, since the tool sits far from the
    # camera's own origin.
    R_delta = Rotation.from_euler('xyz', rot_deg, degrees=True).as_matrix()
    matrix = numpy.eye(4)
    matrix[0:3, 0:3] = R_delta
    if pivot is None:
        matrix[0:3, 3] = trans_m
    else:
        pivot = numpy.asarray(pivot)
        matrix[0:3, 3] = (numpy.eye(3) - R_delta) @ pivot + numpy.asarray(trans_m)
    return matrix


def slider_to_value(pos: int, rng: float) -> float:
    return (pos - SLIDER_CENTER) / SLIDER_CENTER * rng


def value_to_slider(value: float, rng: float) -> int:
    if rng == 0:
        return SLIDER_CENTER
    pos = int(round(value / rng * SLIDER_CENTER)) + SLIDER_CENTER
    return max(0, min(SLIDER_MAX, pos))


def average_transforms(matrices):
    # matrices: list of 4x4 cam_T_base estimates, one per saved correction --
    # each is one noisy sample of the same unknown fixed calibration. Same
    # averaging as 3d_pose_alignment.py's average_transforms: translation by
    # arithmetic mean, rotation by SVD orthonormalization of the mean
    # rotation matrix (the "chordal L2 mean").
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


def write_hand_eye_calibration(json_file: str, cam_T_base: numpy.ndarray, history_entry: dict = None) -> str:
    with open(json_file, 'r') as f:
        data = json.load(f)
    # base-frame.transform stays the calibration everything renders with (now
    # post-alignment). The first correction applied on top of a fresh
    # camera_registration.py result (which writes only base-frame) also
    # keeps that raw result as 'pre-alignment' -- later corrections never
    # overwrite it, so it's always the un-aligned hand-eye calibration.
    if 'pre-alignment' not in data:
        data['pre-alignment'] = {
            'reference-frame': data['base-frame'].get('reference-frame'),
            'transform': data['base-frame']['transform'],
            'recorded_at': time.time(),
        }
    data['base-frame']['transform'] = cam_T_base.tolist()
    if history_entry is not None:
        data.setdefault('alignment-history', []).append(history_entry)

    backup_path = f'{json_file}.bak-{int(time.time())}'
    shutil.copy2(json_file, backup_path)
    with open(json_file, 'w') as f:
        json.dump(data, f)
    return backup_path


def maybe_apply_hand_eye_correction(hand_eye_json: str, cam_T_base_at_capture: numpy.ndarray, records: list,
                                    dataset_dir: str = None):
    # cam_T_base_at_capture: the calibration that was IN EFFECT when this
    # dataset was captured (from the dataset's own meta) -- each saved
    # frame's manual_correction.delta_T is a correction ON TOP OF that, same
    # as 3d_pose_alignment.py's cam_T_base_estimate.
    C_matrices = [
        numpy.array(r['manual_correction']['delta_T']).reshape((4, 4)) @ cam_T_base_at_capture
        for r in records if r.get('alignment_status') == 'saved'
    ]
    if not C_matrices:
        print('No saved manual corrections in this dataset -- nothing to apply to the hand-eye calibration.')
        return

    refined, diag = average_transforms(C_matrices)
    print(f"\n{len(C_matrices)} saved correction(s) out of {len(records)} frame(s). "
          f"Spread: {diag['trans_std_mm']:.2f}mm / {diag['rot_std_deg']:.2f}deg std "
          f"(max {diag['trans_max_mm']:.2f}mm / {diag['rot_max_deg']:.2f}deg).")

    if not os.path.exists(hand_eye_json):
        print(f'{hand_eye_json} does not exist -- cannot apply the correction to it.')
        return

    current = load_hand_eye_calibration(hand_eye_json)
    trans_diff_mm = numpy.linalg.norm(refined[0:3, 3] - current[0:3, 3]) * 1000.0
    rot_diff_deg = Rotation.from_matrix(current[0:3, 0:3].T @ refined[0:3, 0:3]).magnitude() * 180.0 / numpy.pi
    print(f'Applying this would change {hand_eye_json} by {trans_diff_mm:.2f}mm / {rot_diff_deg:.2f}deg.')

    answer = input('Apply this correction to the hand-eye calibration file? [y/N]: ').strip().lower()
    if answer != 'y':
        print('Not updating the hand-eye calibration file.')
        return

    history_entry = {
        'timestamp': time.time(),
        'dataset': os.path.abspath(dataset_dir) if dataset_dir else None,
        'num_frames_saved': len(C_matrices),
        'num_frames': len(records),
        'cam_T_base_at_capture': cam_T_base_at_capture.tolist(),
        'previous_transform': current.tolist(),
        'new_transform': refined.tolist(),
        'change_mm': float(trans_diff_mm),
        'change_deg': float(rot_diff_deg),
        'spread': diag,
    }
    backup_path = write_hand_eye_calibration(hand_eye_json, refined, history_entry)
    print(f'Updated {hand_eye_json} (previous version backed up to {backup_path}).')


# ---------------------------------------------------------------------------
# phase 1: capture
# ---------------------------------------------------------------------------

class RawCapture:
    def __init__(self, node, arm_handle, cam_T_base: numpy.ndarray, out_dir: str, num_frames: int,
                 startup_delay: float, max_pose_age: float,
                 marker_size: float, jaw_length_m: float, jaw_half_angle_deg: float, jaw_half_width_m: float):
        self.node = node
        self.arm_handle = arm_handle
        self.cam_T_base = cam_T_base
        self.out_dir = out_dir
        self.num_frames = num_frames
        self.startup_delay = startup_delay
        self.max_pose_age = max_pose_age
        self.marker_size = marker_size
        self.jaw_length_m = jaw_length_m
        self.jaw_half_angle_deg = jaw_half_angle_deg
        self.jaw_half_width_m = jaw_half_width_m
        self.start_time = time.time()

        os.makedirs(self.out_dir, exist_ok=True)
        for sub in IMAGE_SUBDIRS:
            os.makedirs(os.path.join(self.out_dir, sub), exist_ok=True)

        self.dataset_json_path = os.path.join(self.out_dir, DATASET_JSON_NAME)
        data = {}
        if os.path.exists(self.dataset_json_path):
            with open(self.dataset_json_path, 'r') as f:
                data = json.load(f)
        self.records = data.get('records', [])
        self.meta = data.get('meta', {})
        self.count = len(self.records)

        self.bridge = cv_bridge.CvBridge()
        self.dist_coeffs = numpy.zeros((5, 1))

        self.latest_pose = {}
        self.latest_camera_info = {}
        self.camera_matrix = None
        self._pending = None  # latest synced (left, right, right_rect) image triple

        left_topic = f'/{RIG}/left/image_raw'
        right_topic = f'/{RIG}/right/image_raw'
        right_rect_topic = f'/{RIG}/right/image_rect'
        left_info_topic = f'/{RIG}/left/camera_info'
        right_info_topic = f'/{RIG}/right/camera_info'

        print(f'Subscribing to {left_topic}, {right_topic}, {right_rect_topic}')
        print(f'Subscribing to {left_info_topic}, {right_info_topic}')
        for name, topic in POSE_TOPICS.items():
            print(f'Subscribing to {topic} ({name})')
        print(f'Using PSM1 crtk handle for {GRIPPER_POSE_NAME} and {SHAFT_POSE_NAME} (forward kinematics)')
        print(f'Saving dataset to ./{self.out_dir}/ ({self.count}/{self.num_frames} frames already captured)')

        self.node.create_subscription(
            sensor_msgs.msg.CameraInfo, left_info_topic, self._info_callback('left'), 10
        )
        self.node.create_subscription(
            sensor_msgs.msg.CameraInfo, right_info_topic, self._info_callback('right'), 10
        )
        for name, topic in POSE_TOPICS.items():
            self.node.create_subscription(
                geometry_msgs.msg.PoseStamped, topic, self._pose_callback(name), 10
            )

        left_sub = message_filters.Subscriber(self.node, sensor_msgs.msg.Image, left_topic)
        right_sub = message_filters.Subscriber(self.node, sensor_msgs.msg.Image, right_topic)
        right_rect_sub = message_filters.Subscriber(self.node, sensor_msgs.msg.Image, right_rect_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub, right_rect_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self._image_callback)

    # -- subscriptions --------------------------------------------------

    def _info_callback(self, name):
        def cb(info_msg):
            self.latest_camera_info[name] = (info_msg, time.time())
            if name == 'right':
                projection_matrix = numpy.array(info_msg.p).reshape((3, 4))
                self.camera_matrix = projection_matrix[0:3, 0:3]
        return cb

    def _pose_callback(self, name):
        def cb(msg):
            self.latest_pose[name] = (msg, time.time())
        return cb

    def _image_callback(self, left_msg, right_msg, right_rect_msg):
        # Keep this cheap -- runs on crtk's single-threaded executor. Capture
        # (which needs a blocking forward_kinematics call, only safe from the
        # main thread) happens in try_capture() below, not here.
        self._pending = (left_msg, right_msg, right_rect_msg)

    # -- persistence ------------------------------------------------------

    def _save_json(self):
        meta = dict(self.meta)
        meta['rig'] = RIG
        meta['cam_T_base'] = self.cam_T_base.tolist()
        meta['cam_T_base_rodrigues'] = matrix_to_rodrigues(self.cam_T_base)
        meta['gripper_pose_name'] = GRIPPER_POSE_NAME
        meta['shaft_pose_name'] = SHAFT_POSE_NAME
        meta['pose_topics'] = POSE_TOPICS
        meta['num_frames_target'] = self.num_frames
        if 'left' in self.latest_camera_info:
            meta['left_camera_info'] = camera_info_record(self.latest_camera_info['left'][0])
        if 'right' in self.latest_camera_info:
            meta['right_camera_info'] = camera_info_record(self.latest_camera_info['right'][0])
        self.meta = meta
        with open(self.dataset_json_path, 'w') as f:
            json.dump({'meta': meta, 'records': self.records}, f, indent=2)

    # -- readiness --------------------------------------------------------

    def is_done(self) -> bool:
        return self.count >= self.num_frames

    def _missing_or_stale(self, now):
        missing = []
        for name in POSE_TOPICS:
            entry = self.latest_pose.get(name)
            if entry is None:
                missing.append(f'{name} (not received)')
            elif now - entry[1] > self.max_pose_age:
                missing.append(f'{name} (stale, {now - entry[1]:.2f}s old)')
        for name in ('left', 'right'):
            if self.latest_camera_info.get(name) is None:
                missing.append(f'{name}_camera_info (not received)')
        return missing

    # -- live preview -------------------------------------------------

    def update_live(self):
        # Assumes the caller has already created LIVE_WINDOW_NAME.
        if self._pending is None:
            return
        _left_msg, _right_msg, right_rect_msg = self._pending
        img = self.bridge.imgmsg_to_cv2(right_rect_msg, desired_encoding='bgr8').copy()

        now = time.time()
        lines = [f'Captured {self.count}/{self.num_frames}  --  [c]apture  [q]uit & align  [ESC]abort']
        if now - self.start_time < self.startup_delay:
            lines.append(f'Waiting for startup delay ({self.startup_delay - (now - self.start_time):.1f}s left)')
        else:
            missing = self._missing_or_stale(now)
            lines.append('Not ready: ' + ', '.join(missing) if missing else 'Ready to capture')

        for i, line in enumerate(lines):
            y = 25 + i * 25
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(LIVE_WINDOW_NAME, img)

    def try_capture(self) -> bool:
        if self._pending is None:
            print('No synced frame available yet.')
            return False
        now = time.time()
        if now - self.start_time < self.startup_delay:
            print('Still within startup delay.')
            return False
        if self.camera_matrix is None:
            print('No camera_info received yet for right/camera_info.')
            return False
        missing = self._missing_or_stale(now)
        if missing:
            print(f'Cannot capture, missing/stale data: {", ".join(missing)}')
            return False

        left_msg, right_msg, right_rect_msg = self._pending
        try:
            psm1_cp, _ = self.arm_handle.measured_cp()
            psm1_local_cp, _ = self.arm_handle.local.measured_cp()
            psm1_jp, _ = self.arm_handle.measured_jp()
            shaft_end_local_cp = self.arm_handle.local.forward_kinematics(psm1_jp[0:4])
        except TimeoutError as e:
            print(f'Cannot capture, PSM1 pose/kinematics not available: {e}')
            return False

        poses = {
            name: pose_record(msg, recv_time, now)
            for name, (msg, recv_time) in self.latest_pose.items()
        }
        poses['psm1_measured_cp'] = kdl_pose_record(psm1_cp, now)
        poses[GRIPPER_POSE_NAME] = kdl_pose_record(psm1_local_cp, now)
        poses[SHAFT_POSE_NAME] = kdl_pose_record(shaft_end_local_cp, now)

        base_T_gripper = kdl_frame_to_matrix(psm1_local_cp)
        base_T_shaft_end = kdl_frame_to_matrix(shaft_end_local_cp)
        cam_T_gripper_nominal = self.cam_T_base @ base_T_gripper
        cam_T_shaft_end_nominal = self.cam_T_base @ base_T_shaft_end

        idx = self.count
        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        right_rect_img = self.bridge.imgmsg_to_cv2(right_rect_msg, desired_encoding='bgr8')

        lfname = f'left-{idx:04d}.png'
        rfname = f'right-{idx:04d}.png'
        rectfname = f'right_rect-{idx:04d}.png'
        nominal_overlay_fname = f'overlay-{idx:04d}.png'
        cv2.imwrite(os.path.join(self.out_dir, 'left', lfname), left_img)
        cv2.imwrite(os.path.join(self.out_dir, 'right', rfname), right_img)
        cv2.imwrite(os.path.join(self.out_dir, 'right_rect', rectfname), right_rect_img)

        nominal_overlay_img, _, _ = draw_full_overlay(
            right_rect_img.copy(), self.camera_matrix, self.dist_coeffs,
            cam_T_shaft_end_nominal, cam_T_gripper_nominal,
            self.marker_size, self.jaw_length_m, self.jaw_half_angle_deg, self.jaw_half_width_m,
        )
        cv2.imwrite(os.path.join(self.out_dir, 'overlay_nominal', nominal_overlay_fname), nominal_overlay_img)

        self.records.append({
            'frame_index': idx,
            'timestamp': now,
            'left_image': f'left/{lfname}',
            'right_image': f'right/{rfname}',
            'right_rect_image': f'right_rect/{rectfname}',
            'overlay_nominal_image': f'overlay_nominal/{nominal_overlay_fname}',
            'poses': poses,
            'cam_T_gripper_nominal': cam_T_gripper_nominal.tolist(),
            'cam_T_shaft_end_nominal': cam_T_shaft_end_nominal.tolist(),
            'alignment_status': 'pending',
        })
        self.count += 1
        self._save_json()
        print(f'Captured frame {idx} ({self.count}/{self.num_frames}): {lfname}, {rfname}, {rectfname}')
        return True


# ---------------------------------------------------------------------------
# phase 2: manual alignment (mirrors 3d_pose_alignment.py's AlignmentGUI, but
# writes the manual pose straight back into each frame's own record instead
# of a separate corrections file, and never touches the hand-eye calibration)
# ---------------------------------------------------------------------------

class AlignmentPass:
    def __init__(self, out_dir: str, dataset_json_path: str, meta: dict, records: list,
                 camera_matrix: numpy.ndarray, dist_coeffs: numpy.ndarray,
                 marker_size: float, jaw_length_m: float, jaw_half_angle_deg: float, jaw_half_width_m: float,
                 trans_range_m: float, rot_range_deg: float):
        self.out_dir = out_dir
        self.dataset_json_path = dataset_json_path
        self.meta = meta
        self.records = records
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size
        self.jaw_length_m = jaw_length_m
        self.jaw_half_angle_deg = jaw_half_angle_deg
        self.jaw_half_width_m = jaw_half_width_m
        self.base_trans_range_m = trans_range_m
        self.base_rot_range_deg = rot_range_deg
        self.range_multiplier = 1

        self.current_trans_m = [0.0, 0.0, 0.0]
        self.current_rot_deg = [0.0, 0.0, 0.0]
        self._drag_point = None
        self._drag_last_xy = None
        self._last_gripper_origin = None
        self._last_gripper_depth = None
        self._last_shaft_origin = None
        self._last_shaft_depth = None
        self._last_delta_T = None
        self._last_cam_T_gripper_corrected = None
        self._last_cam_T_shaft_corrected = None
        self._base_img = None

        start_index = next(
            (i for i, r in enumerate(records) if r.get('alignment_status', 'pending') == 'pending'), 0
        )

        cv2.namedWindow(ALIGN_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Image', ALIGN_WINDOW_NAME, start_index, max(len(records) - 1, 0),
                            self._on_image_trackbar)
        cv2.createTrackbar('Range x', ALIGN_WINDOW_NAME, 0, RANGE_STEPS - 1, self._on_range_trackbar)
        cv2.createTrackbar('Trans X', ALIGN_WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Trans Y', ALIGN_WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Trans Z', ALIGN_WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Rot X', ALIGN_WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Rot Y', ALIGN_WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.createTrackbar('Rot Z', ALIGN_WINDOW_NAME, SLIDER_CENTER, SLIDER_MAX, self._on_pose_trackbar)
        cv2.setMouseCallback(ALIGN_WINDOW_NAME, self._on_mouse)

        self.current_index = -1
        self._load_frame(start_index)

    def _on_image_trackbar(self, pos):
        self._load_frame(pos)

    def _on_range_trackbar(self, pos):
        self.range_multiplier = pos + 1
        self._set_trackbars_from_state()

    def _on_pose_trackbar(self, _pos):
        trans_range_m, rot_range_deg = self._effective_ranges()
        self.current_trans_m = [
            slider_to_value(cv2.getTrackbarPos('Trans X', ALIGN_WINDOW_NAME), trans_range_m),
            slider_to_value(cv2.getTrackbarPos('Trans Y', ALIGN_WINDOW_NAME), trans_range_m),
            slider_to_value(cv2.getTrackbarPos('Trans Z', ALIGN_WINDOW_NAME), trans_range_m),
        ]
        self.current_rot_deg = [
            slider_to_value(cv2.getTrackbarPos('Rot X', ALIGN_WINDOW_NAME), rot_range_deg),
            slider_to_value(cv2.getTrackbarPos('Rot Y', ALIGN_WINDOW_NAME), rot_range_deg),
            slider_to_value(cv2.getTrackbarPos('Rot Z', ALIGN_WINDOW_NAME), rot_range_deg),
        ]
        self._render()

    def _effective_ranges(self):
        return (self.base_trans_range_m * self.range_multiplier,
                self.base_rot_range_deg * self.range_multiplier)

    def _load_frame(self, index):
        self.current_index = index
        record = self.records[index]
        saved = record.get('manual_correction')
        self.current_trans_m = list(saved['trans_m']) if saved else [0.0, 0.0, 0.0]
        self.current_rot_deg = list(saved['rot_deg']) if saved else [0.0, 0.0, 0.0]

        image_path = os.path.join(self.out_dir, record['right_rect_image'])
        self._base_img = cv2.imread(image_path)
        if self._base_img is None:
            print(f'Could not read {image_path}')

        self._widen_range_if_needed()
        self._set_trackbars_from_state()

    def _widen_range_if_needed(self):
        needed = 1
        for value in self.current_trans_m:
            if self.base_trans_range_m > 0:
                needed = max(needed, int(numpy.ceil(abs(value) / self.base_trans_range_m)))
        for value in self.current_rot_deg:
            if self.base_rot_range_deg > 0:
                needed = max(needed, int(numpy.ceil(abs(value) / self.base_rot_range_deg)))
        needed = min(needed, RANGE_STEPS)
        self.range_multiplier = needed
        cv2.setTrackbarPos('Range x', ALIGN_WINDOW_NAME, needed - 1)  # triggers _on_range_trackbar

    def _set_trackbars_from_state(self):
        trans_range_m, rot_range_deg = self._effective_ranges()
        cv2.setTrackbarPos('Trans X', ALIGN_WINDOW_NAME, value_to_slider(self.current_trans_m[0], trans_range_m))
        cv2.setTrackbarPos('Trans Y', ALIGN_WINDOW_NAME, value_to_slider(self.current_trans_m[1], trans_range_m))
        cv2.setTrackbarPos('Trans Z', ALIGN_WINDOW_NAME, value_to_slider(self.current_trans_m[2], trans_range_m))
        cv2.setTrackbarPos('Rot X', ALIGN_WINDOW_NAME, value_to_slider(self.current_rot_deg[0], rot_range_deg))
        cv2.setTrackbarPos('Rot Y', ALIGN_WINDOW_NAME, value_to_slider(self.current_rot_deg[1], rot_range_deg))
        cv2.setTrackbarPos('Rot Z', ALIGN_WINDOW_NAME, value_to_slider(self.current_rot_deg[2], rot_range_deg))
        self._render()

    def _sync_trans_sliders(self):
        trans_range_m, _ = self._effective_ranges()
        cv2.setTrackbarPos('Trans X', ALIGN_WINDOW_NAME, value_to_slider(self.current_trans_m[0], trans_range_m))
        cv2.setTrackbarPos('Trans Y', ALIGN_WINDOW_NAME, value_to_slider(self.current_trans_m[1], trans_range_m))
        cv2.setTrackbarPos('Trans Z', ALIGN_WINDOW_NAME, value_to_slider(self.current_trans_m[2], trans_range_m))

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
        depth = self._last_gripper_depth if self._drag_point == 'gripper' else self._last_shaft_depth
        if depth is None:
            return
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        self.current_trans_m[0] += dx_px * depth / fx
        self.current_trans_m[1] += dy_px * depth / fy
        self._widen_range_if_needed()
        self._sync_trans_sliders()
        self._render()

    def _render(self):
        record = self.records[self.current_index]
        if self._base_img is None:
            return
        img = self._base_img.copy()

        cam_T_gripper_nominal = numpy.array(record['cam_T_gripper_nominal']).reshape((4, 4))
        cam_T_shaft_nominal = numpy.array(record['cam_T_shaft_end_nominal']).reshape((4, 4))
        pivot = cam_T_gripper_nominal[0:3, 3]
        delta_T = delta_transform(self.current_trans_m, self.current_rot_deg, pivot=pivot)
        cam_T_gripper_corrected = delta_T @ cam_T_gripper_nominal
        cam_T_shaft_corrected = delta_T @ cam_T_shaft_nominal

        img, shaft_origin, gripper_origin = draw_full_overlay(
            img, self.camera_matrix, self.dist_coeffs, cam_T_shaft_corrected, cam_T_gripper_corrected,
            self.marker_size, self.jaw_length_m, self.jaw_half_angle_deg, self.jaw_half_width_m,
        )
        self._last_shaft_origin = shaft_origin
        self._last_shaft_depth = float(cam_T_shaft_corrected[2, 3])
        self._last_gripper_origin = gripper_origin
        self._last_gripper_depth = float(cam_T_gripper_corrected[2, 3])
        self._last_delta_T = delta_T
        self._last_cam_T_gripper_corrected = cam_T_gripper_corrected
        self._last_cam_T_shaft_corrected = cam_T_shaft_corrected

        status = record.get('alignment_status', 'pending').upper()
        lines = [
            f'frame {record["frame_index"]}  ({self.current_index + 1}/{len(self.records)})  [{status}]',
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
        cv2.imshow(ALIGN_WINDOW_NAME, img)

    def _save_json(self):
        with open(self.dataset_json_path, 'w') as f:
            json.dump({'meta': self.meta, 'records': self.records}, f, indent=2)

    def _clear_manual_fields(self, record):
        for key in ('manual_correction', 'cam_T_gripper_manual', 'cam_T_shaft_end_manual',
                    'manual_pose_rodrigues', 'manual_pose_shaft_end_rodrigues', 'overlay_manual_image'):
            record.pop(key, None)

    def _save_current(self):
        record = self.records[self.current_index]
        record['manual_correction'] = {
            'trans_m': list(self.current_trans_m),
            'rot_deg': list(self.current_rot_deg),
            'delta_T': self._last_delta_T.tolist(),
        }
        record['cam_T_gripper_manual'] = self._last_cam_T_gripper_corrected.tolist()
        record['cam_T_shaft_end_manual'] = self._last_cam_T_shaft_corrected.tolist()
        # primary manual 6D pose (gripper tip) in camera frame, Rodrigues tx,ty,tz,rx,ry,rz
        record['manual_pose_rodrigues'] = matrix_to_rodrigues(self._last_cam_T_gripper_corrected)
        record['manual_pose_shaft_end_rodrigues'] = matrix_to_rodrigues(self._last_cam_T_shaft_corrected)
        record['alignment_status'] = 'saved'

        overlay_fname = f'overlay-{record["frame_index"]:04d}.png'
        overlay_img, _, _ = draw_full_overlay(
            self._base_img.copy(), self.camera_matrix, self.dist_coeffs,
            self._last_cam_T_shaft_corrected, self._last_cam_T_gripper_corrected,
            self.marker_size, self.jaw_length_m, self.jaw_half_angle_deg, self.jaw_half_width_m,
        )
        cv2.imwrite(os.path.join(self.out_dir, 'overlay_manual', overlay_fname), overlay_img)
        record['overlay_manual_image'] = f'overlay_manual/{overlay_fname}'

        self._save_json()
        print(f'Saved manual alignment for frame {record["frame_index"]}: '
              f'trans(mm)={[round(v*1000, 2) for v in self.current_trans_m]} '
              f'rot(deg)={[round(v, 2) for v in self.current_rot_deg]}')
        self._render()

    def _reset_current(self):
        record = self.records[self.current_index]
        self._clear_manual_fields(record)
        record['alignment_status'] = 'pending'
        self.current_trans_m = [0.0, 0.0, 0.0]
        self.current_rot_deg = [0.0, 0.0, 0.0]
        self.range_multiplier = 1
        cv2.setTrackbarPos('Range x', ALIGN_WINDOW_NAME, 0)  # triggers _on_range_trackbar -> _set_trackbars_from_state
        self._set_trackbars_from_state()
        self._save_json()

    def _skip_current(self):
        record = self.records[self.current_index]
        self._clear_manual_fields(record)
        record['alignment_status'] = 'skipped'
        self._save_json()
        print(f'Skipped frame {record["frame_index"]}.')
        self._go(1)

    def _go(self, delta):
        new_index = max(0, min(len(self.records) - 1, self.current_index + delta))
        cv2.setTrackbarPos('Image', ALIGN_WINDOW_NAME, new_index)  # triggers _load_frame

    def run(self):
        total = len(self.records)
        saved = sum(1 for r in self.records if r.get('alignment_status') == 'saved')
        skipped = sum(1 for r in self.records if r.get('alignment_status') == 'skipped')
        print(f'{total} frame(s) to align, {saved} already saved, {skipped} skipped.')
        print('Drag the yellow (shaft) or red (gripper) dot, or use the Trans/Rot sliders, then press s to save.')
        print("Widen 'Range x' if the tool is too far off to reach at 1x.")
        print('n/p to move between frames, x to skip a frame, q to quit (progress is saved after every s/r/x).')
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
        cv2.destroyAllWindows()


def load_dataset_for_alignment(out_dir: str):
    dataset_json_path = os.path.join(out_dir, DATASET_JSON_NAME)
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f'{dataset_json_path} not found -- capture a dataset first (without --align-only)')
    with open(dataset_json_path, 'r') as f:
        data = json.load(f)
    meta = data.get('meta', {})
    records = data.get('records', [])
    if not records:
        raise ValueError(f'{dataset_json_path} has no recorded frames yet')
    if 'right_camera_info' not in meta:
        raise ValueError(f'{dataset_json_path} has no right_camera_info in its meta -- cannot project poses')
    p_matrix = numpy.array(meta['right_camera_info']['P']).reshape((3, 4))
    camera_matrix = p_matrix[0:3, 0:3]
    dist_coeffs = numpy.zeros((5, 1))
    for sub in IMAGE_SUBDIRS:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    return dataset_json_path, meta, records, camera_matrix, dist_coeffs


def main():
    args = None  # populated below before any ROS calls happen

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-H', '--hand-eye-json', type=str, default=DEFAULT_HAND_EYE_JSON,
        help='hand-eye calibration matrix in JSON format using OpenCV coordinate system '
             f'(default: {DEFAULT_HAND_EYE_JSON})',
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default='hand_eye_rodrigues_dataset',
        help='directory to save the dataset in (created if missing; re-running resumes into it)',
    )
    parser.add_argument(
        '-n', '--num-frames', type=int, default=20,
        help='number of frames to capture (default: 20)',
    )
    parser.add_argument(
        '--align-only', action='store_true',
        help='skip capturing new frames and just run the manual-alignment pass on an existing dataset '
             'directory -- fully offline, no ROS/dVRK connection needed',
    )
    parser.add_argument(
        '--startup-delay', type=float, default=5.0,
        help='seconds to wait after connecting before allowing the first capture',
    )
    parser.add_argument(
        '--max-pose-age', type=float, default=1.0,
        help='maximum age in seconds of a cached pose/camera_info message to still be considered valid',
    )
    parser.add_argument(
        '--trans-range-mm', type=float, default=50.0,
        help='manual-alignment translation slider range in mm at 1x (default: 50mm)',
    )
    parser.add_argument(
        '--rot-range-deg', type=float, default=180.0,
        help='manual-alignment rotation slider range in degrees at 1x (default: 180deg)',
    )
    parser.add_argument(
        '--marker-size', type=float, default=0.01,
        help='axis marker length in meters (default: 0.01)',
    )
    parser.add_argument(
        '--jaw-length-mm', type=float, default=10.0,
        help='length of the drawn jaw blades from the gripper point, in mm (default: 10mm)',
    )
    parser.add_argument(
        '--jaw-angle-deg', type=float, default=12.0,
        help='half-angle between the two drawn jaw blades, in degrees (default: 12deg)',
    )
    parser.add_argument(
        '--jaw-width-mm', type=float, default=4.0,
        help='width of the drawn jaw blades and the shaft-point pivot ring, in mm (default: 4mm)',
    )

    if '--align-only' in sys.argv[1:]:
        args = parser.parse_args(sys.argv[1:])
    else:
        argv = crtk.ral.parse_argv(sys.argv[1:])  # also calls rclpy.init(), skip argv[0]
        args = parser.parse_args(argv)

    jaw_length_m = args.jaw_length_mm / 1000.0
    jaw_half_width_m = args.jaw_width_mm / 1000.0 / 2.0
    trans_range_m = args.trans_range_mm / 1000.0

    if args.align_only:
        dataset_json_path, meta, records, camera_matrix, dist_coeffs = load_dataset_for_alignment(args.output_dir)
        pending = sum(1 for r in records if r.get('alignment_status', 'pending') == 'pending')
        if pending == 0:
            print('All frames are already aligned or skipped -- nothing to do.')
        else:
            align_pass = AlignmentPass(
                args.output_dir, dataset_json_path, meta, records, camera_matrix, dist_coeffs,
                args.marker_size, jaw_length_m, args.jaw_angle_deg, jaw_half_width_m,
                trans_range_m, args.rot_range_deg,
            )
            align_pass.run()
            print(f'Alignment pass done. Dataset saved to ./{args.output_dir}/')

        cam_T_base_at_capture = numpy.array(meta['cam_T_base']).reshape((4, 4))
        maybe_apply_hand_eye_correction(args.hand_eye_json, cam_T_base_at_capture, records, args.output_dir)
        return

    cam_T_base = load_hand_eye_calibration(args.hand_eye_json)

    ral = crtk.ral('hand_eye_rodrigues_dataset')
    arm_handle = dvrk_camera_registration.ARM(ral, arm_name='PSM1', expected_interval=0.5)
    ral.spin()
    ral.check_connections()

    capture = RawCapture(
        ral._node, arm_handle, cam_T_base, args.output_dir, args.num_frames,
        args.startup_delay, args.max_pose_age, args.marker_size, jaw_length_m, args.jaw_angle_deg, jaw_half_width_m,
    )

    if capture.is_done():
        print(f'Already have {capture.count}/{capture.num_frames} frames captured. '
              f're-run with --align-only to run the alignment pass on them.')
        ral.shutdown()
        return

    print(f'Waiting for synced image pairs... (capture starts in {args.startup_delay:.0f}s)')
    print('Press c to capture a frame, q to stop capturing and move on to alignment, '
          'ESC to abort without aligning.')

    live_window_created = False
    aborted = False
    try:
        while not ral.is_shutdown() and not capture.is_done():
            if not live_window_created:
                cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL)
                live_window_created = True
            capture.update_live()
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC: abort without aligning
                aborted = True
                break
            elif key == ord('q'):  # stop capturing, proceed to alignment
                break
            elif key == ord('c'):
                capture.try_capture()
    except KeyboardInterrupt:
        aborted = True
    if live_window_created:
        cv2.destroyWindow(LIVE_WINDOW_NAME)

    print(f'Captured {capture.count}/{capture.num_frames} frames.')
    ral.shutdown()

    if aborted:
        print(f're-run with --align-only to align the {capture.count} captured frame(s) later.')
        return
    if capture.count == 0:
        print('Nothing captured -- nothing to align.')
        return
    if capture.camera_matrix is None:
        print('No camera_info was received -- re-run with --align-only once camera_info is available.')
        return

    align_pass = AlignmentPass(
        args.output_dir, capture.dataset_json_path, capture.meta, capture.records,
        capture.camera_matrix, capture.dist_coeffs, args.marker_size, jaw_length_m, args.jaw_angle_deg,
        jaw_half_width_m, trans_range_m, args.rot_range_deg,
    )
    align_pass.run()
    print(f'Alignment pass done. Dataset saved to ./{args.output_dir}/')

    cam_T_base_at_capture = numpy.array(capture.meta['cam_T_base']).reshape((4, 4))
    maybe_apply_hand_eye_correction(args.hand_eye_json, cam_T_base_at_capture, capture.records, args.output_dir)


if __name__ == '__main__':
    main()
