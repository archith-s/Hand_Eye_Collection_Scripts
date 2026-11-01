#!/usr/bin/env python3

# Captures synced left/right image pairs (same approach as
# charuco_image_generation.py) together with the dVRK gripper pose at
# capture time, using an existing hand-eye calibration (as loaded by
# dvrk_camera_registration's vis_gripper_pose.py) to also express that
# pose in the camera frame. Produces a dataset of image pairs + a JSON
# file of ground-truth gripper poses for evaluating hand-eye calibration.
# The shaft-end pose (forward kinematics of the first 4 joints, same as
# vis_gripper_pose.py's second axis) is saved too: the CAD model used
# downstream is the wrist clevis, whose pose needs both axes (see
# hand_eye_diff_render_test.py's cam_T_joint_model).
#
# By default a live preview window (rectified right image) is shown and a
# frame is only captured when you press 'c' (q or ESC to quit). Pass
# --automatic to instead capture a frame every --interval seconds.
#
# gripper_ground_truth.json uses the hand-eye calibration file's
# base-frame.transform -- after hand_eye_rodrigues_dataset.py's GUI
# alignment has been applied, that's the post-alignment calibration, the
# primary one everything downstream renders with. Alongside it,
# calibration_variants.json records, per frame, the gripper/shaft-end pose
# in the camera frame three ways, plus the camera intrinsics (shared by
# all three) to project/render any of them:
#
#   kinematic                pure kinematics through the setup joints:
#                            cam <- ECM tip <- World -> PSM RCM -> PSM tip
#                            (/ECM/measured_cp, /SUJ/PSMx/local/measured_cp,
#                            /PSMx/local/measured_cp), with the nominal dVRK
#                            ECM-tip -> OpenCV camera flip. Least accurate.
#   hand_eye_pre_alignment   raw camera_registration.py cam_T_base (the
#                            calibration file's 'pre-alignment' block, or
#                            --pre-alignment-json) @ PSM RCM -> tool.
#   hand_eye_post_alignment  base-frame.transform @ PSM RCM -> tool -- same
#                            poses as gripper_ground_truth.json. Primary.
#
# Images: the raw left/right frames are saved as-is (left-/right-XXXX.png)
# and also stereo-rectified here (left_rect-/right_rect-XXXX.png) exactly
# like OpencvCameraCalibration's stereo_camera_calibration_archits_charuco_board.sh
# does (cv2.initUndistortRectifyMap(K, D, R, P, size, CV_16SC2) + cv2.remap
# INTER_LINEAR, from that run's left.yaml/right.yaml). This is NOT the same
# as the rig's /right/image_rect, which only undistorts (R = I, P = [K|0]).
#
# The hand-eye calibration is relative to the raw (un-rotated) right
# optical frame; the stereo-rectified right camera is rotated from that by
# right.yaml's rectification_matrix R. So every cam_T_* below is
# rect_T_cam_raw @ (raw-frame pose), and projects onto right_rect-XXXX.png
# with P[:3,:3] and zero distortion. The raw-frame poses are kept as
# cam_raw_T_*.

import argparse
import json
import os
import sys
import time

import cv2
import cv_bridge
import geometry_msgs.msg
import message_filters
import numpy
import sensor_msgs.msg
import yaml
from scipy.spatial.transform import Rotation

import crtk
import dvrk_camera_registration

from hand_eye_rodrigues_dataset import camera_info_record, kdl_pose_record, pose_record

RIG = os.environ.get('RIG', 'jhu_dVRK')  # falls back if $RIG isn't set

LIVE_WINDOW_NAME = 'Ground truth capture (live)'

# camera_registration.py always writes PSM1's hand-eye calibration here
DEFAULT_HAND_EYE_JSON = os.path.expanduser('~/PSM1-registration-open-cv.json')

# output_path of stereo_camera_calibration_archits_charuco_board.sh
DEFAULT_STEREO_CALIB_DIR = os.path.expanduser(
    '~/OpencvCameraCalibration/output/charuco-test-dataset/results_with_full_dist_dvrk_classic')

VARIANTS_JSON_NAME = 'calibration_variants.json'
PRIMARY_VARIANT = 'hand_eye_post_alignment'

# Nominal dVRK ECM tip frame -> OpenCV camera frame: the same x/y flip
# camera_registration.py's save_registration applies to go from OpenCV to
# dVRK convention (it's its own inverse). Ignores the real tip -> optical
# centre offset, which is part of why the kinematic variant is least accurate.
CAM_T_ECM_TIP = numpy.diag([-1.0, -1.0, 1.0, 1.0])

# overlay colors (BGR) per variant
VARIANT_COLORS = {
    'kinematic': (0, 140, 255),
    'hand_eye_pre_alignment': (255, 0, 255),
    'hand_eye_post_alignment': (0, 255, 0),
}


def suj_pose_topics(psm_name: str) -> dict:
    # name -> topic for the setup-joint/ECM half of the kinematic chain (the
    # PSM's own measured_cp comes from the crtk ARM handle instead)
    return {
        'ecm_measured_cp': '/ECM/measured_cp',                         # World -> ECM tip
        'ecm_local_measured_cp': '/ECM/local/measured_cp',             # ECM RCM -> ECM tip
        'suj_ecm_measured_cp': '/SUJ/ECM/measured_cp',                 # World -> ECM RCM
        'suj_ecm_local_measured_cp': '/SUJ/ECM/local/measured_cp',     # World -> ECM RCM
        'suj_psm_measured_cp': f'/SUJ/{psm_name}/measured_cp',         # ECM tip -> PSM RCM
        'suj_psm_local_measured_cp': f'/SUJ/{psm_name}/local/measured_cp',  # World -> PSM RCM
    }


def load_hand_eye_calibration(json_file: str) -> numpy.ndarray:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return numpy.array(data['base-frame']['transform']).reshape(4, 4)


def load_calibration_variants(hand_eye_json: str, pre_alignment_json: str = None) -> dict:
    # post = base-frame.transform (what everything renders with); pre = the
    # raw hand-eye result hand_eye_rodrigues_dataset.py preserves the first
    # time it applies a correction, unless overridden by another file
    # (e.g. one of its .bak-* backups) whose base-frame is the raw result.
    with open(hand_eye_json, 'r') as f:
        data = json.load(f)
    post = numpy.array(data['base-frame']['transform']).reshape(4, 4)
    if pre_alignment_json:
        pre = load_hand_eye_calibration(pre_alignment_json)
        pre_source = os.path.abspath(pre_alignment_json)
    elif 'pre-alignment' in data:
        pre = numpy.array(data['pre-alignment']['transform']).reshape(4, 4)
        pre_source = f'{os.path.abspath(hand_eye_json)}:pre-alignment'
    else:
        print(f'WARNING: {hand_eye_json} has no pre-alignment block (no GUI correction applied since the last '
              f'camera_registration.py run, or applied before that was tracked) -- recording '
              f'hand_eye_pre_alignment = post-alignment. Pass --pre-alignment-json to use another file.')
        pre = post.copy()
        pre_source = None
    return {
        'post': post,
        'pre': pre,
        'pre_source': pre_source,
        'alignment_applied': not numpy.allclose(pre, post),
        'alignment_history': data.get('alignment-history', []),
    }


def load_stereo_rectification(calib_dir: str) -> dict:
    # Reads the ROS camera_info-style left.yaml/right.yaml that
    # cam_calib_lib's stereo_calibration_pipeline writes, and builds the same
    # rectification maps its compute_rectification_maps does.
    sides = {}
    for side in ('left', 'right'):
        path = os.path.join(calib_dir, f'{side}.yaml')
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        size = (int(data['image_width']), int(data['image_height']))
        K = numpy.array(data['camera_matrix']['data'], dtype=numpy.float64).reshape(3, 3)
        D = numpy.array(data['distortion_coefficients']['data'], dtype=numpy.float64)
        R = numpy.array(data['rectification_matrix']['data'], dtype=numpy.float64).reshape(3, 3)
        P = numpy.array(data['projection_matrix']['data'], dtype=numpy.float64).reshape(3, 4)
        if numpy.allclose(R, numpy.eye(3)):
            print(f'WARNING: {path} has an identity rectification_matrix -- that\'s an undistort-only '
                  f'(e.g. dvrk_handeye/) yaml, not the stereo rectification.')
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, size, cv2.CV_16SC2)
        sides[side] = {'path': os.path.abspath(path), 'size': size, 'K': K, 'D': D, 'R': R, 'P': P,
                       'map1': map1, 'map2': map2}
    rect_T_cam_raw = numpy.eye(4)
    rect_T_cam_raw[0:3, 0:3] = sides['right']['R']
    return {
        'calib_dir': os.path.abspath(calib_dir),
        'sides': sides,
        'camera_matrix_right_rect': sides['right']['P'][0:3, 0:3].copy(),
        'rect_T_cam_raw': rect_T_cam_raw,
    }


def rectify(rectification: dict, side: str, img: numpy.ndarray) -> numpy.ndarray:
    s = rectification['sides'][side]
    if (img.shape[1], img.shape[0]) != s['size']:
        raise ValueError(f'{side} image is {img.shape[1]}x{img.shape[0]} but {s["path"]} is for '
                         f'{s["size"][0]}x{s["size"][1]}')
    return cv2.remap(img, s['map1'], s['map2'], cv2.INTER_LINEAR)


def stereo_side_record(s: dict) -> dict:
    return {
        'yaml': s['path'],
        'width': s['size'][0],
        'height': s['size'][1],
        'K': s['K'].tolist(),
        'D': s['D'].tolist(),
        'R': s['R'].tolist(),
        'P': s['P'].tolist(),
    }


def project_origin(camera_matrix, pose):
    # pixel of a camera-frame pose's origin, or None if behind the camera
    point = pose[0:3, 3]
    if point[2] <= 0:
        return None
    uvw = camera_matrix @ point
    return [float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])]


def measured_cp_to_matrix(m_cp) -> numpy.ndarray:
    rotation = Rotation.from_quat(m_cp.M.GetQuaternion())
    matrix = numpy.eye(4)
    matrix[0:3, 0:3] = numpy.float64(rotation.as_matrix())
    matrix[0, 3] = m_cp.p[0]
    matrix[1, 3] = m_cp.p[1]
    matrix[2, 3] = m_cp.p[2]
    return matrix


def draw_axis(img, camera_matrix, dist_coeffs, pose, size=0.01):
    # Mirrors dvrk_camera_registration's vis_gripper_pose.py draw_axis, so the
    # overlay reflects the exact same projection math that script uses.
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


class GroundTruthCapture:
    def __init__(self, ral, arm_handle, calibration: dict, rectification: dict, out_dir: str, interval: float,
                 startup_delay: float = 5.0, automatic: bool = False, psm_name: str = 'PSM1',
                 hand_eye_json: str = DEFAULT_HAND_EYE_JSON, max_pose_age: float = 1.0):
        self.arm_handle = arm_handle
        self.calibration = calibration
        self.cam_T_base = calibration['post']  # raw right optical frame
        self.rectification = rectification
        self.rect_T_cam_raw = rectification['rect_T_cam_raw']
        self.psm_name = psm_name
        self.hand_eye_json = hand_eye_json
        self.max_pose_age = max_pose_age
        self.pose_topics = suj_pose_topics(psm_name)
        self.latest_pose = {}
        self.latest_camera_info = {}
        self.out_dir = out_dir
        self.interval = interval
        self.automatic = automatic
        self._pending = None  # latest synced (left, right) raw messages
        self.startup_delay = startup_delay
        self.start_time = time.time()
        os.makedirs(self.out_dir, exist_ok=True)

        self.gt_json_path = os.path.join(self.out_dir, 'gripper_ground_truth.json')
        self.records = self._load_existing_records()
        self.count = len(self.records)
        if self.records and 'camera_matrix_right_rect' not in self.records[0]:
            # older datasets' right_rect-*.png are the rig's undistort-only
            # image_rect, in a different camera frame than these -- don't mix
            raise SystemExit(f'{self.gt_json_path} was recorded before stereo rectification was added '
                             f'(different right_rect frame/intrinsics) -- use a new --output-dir.')
        self.variants_json_path = os.path.join(self.out_dir, VARIANTS_JSON_NAME)
        self.variant_records = self._load_existing_variant_records()
        self.last_save = 0.0

        self.bridge = cv_bridge.CvBridge()
        # stereo-rectified right image: P's 3x3, zero distortion
        self.camera_matrix = rectification['camera_matrix_right_rect']
        self.dist_coeffs = numpy.zeros((5, 1))

        left_topic = f'/{RIG}/left/image_raw'
        right_topic = f'/{RIG}/right/image_raw'
        # camera_info is only recorded (and compared against the stereo
        # calibration's K) -- everything is projected with the stereo
        # rectification loaded above, not with these.
        right_info_topic = f'/{RIG}/right/camera_info'
        left_info_topic = f'/{RIG}/left/camera_info'
        print(f'Subscribing to {left_topic}, {right_topic}, {right_info_topic}, {left_info_topic}')
        print(f'Stereo rectification from {rectification["calib_dir"]}')
        for name, topic in self.pose_topics.items():
            print(f'Subscribing to {topic} ({name})')
        print(f'Saving pairs + gripper ground truth to ./{self.out_dir}/ '
              f'(+ {VARIANTS_JSON_NAME}: kinematic / pre / post-alignment poses)')

        # crtk.ral has no public accessor for its underlying rclpy node,
        # but message_filters needs a real node to subscribe on.
        node = ral._node
        node.create_subscription(
            sensor_msgs.msg.CameraInfo, right_info_topic, self._info_callback, 10
        )
        node.create_subscription(
            sensor_msgs.msg.CameraInfo, left_info_topic, self._left_info_callback, 10
        )
        for name, topic in self.pose_topics.items():
            node.create_subscription(
                geometry_msgs.msg.PoseStamped, topic, self._pose_callback(name), 10
            )
        left_sub = message_filters.Subscriber(node, sensor_msgs.msg.Image, left_topic)
        right_sub = message_filters.Subscriber(node, sensor_msgs.msg.Image, right_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [left_sub, right_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

    def _info_callback(self, info_msg):
        self.latest_camera_info['right'] = info_msg

    def _left_info_callback(self, info_msg):
        self.latest_camera_info['left'] = info_msg

    def _pose_callback(self, name):
        def cb(msg):
            self.latest_pose[name] = (msg, time.time())
        return cb

    def _missing_or_stale(self, now):
        missing = []
        for name in self.pose_topics:
            entry = self.latest_pose.get(name)
            if entry is None:
                missing.append(f'{name} (not received)')
            elif now - entry[1] > self.max_pose_age:
                missing.append(f'{name} (stale, {now - entry[1]:.2f}s old)')
        return missing

    def _load_existing_records(self):
        if os.path.exists(self.gt_json_path):
            with open(self.gt_json_path, 'r') as f:
                return json.load(f)
        return []

    def _load_existing_variant_records(self):
        if os.path.exists(self.variants_json_path):
            with open(self.variants_json_path, 'r') as f:
                return json.load(f).get('records', [])
        return []

    def _save_records(self):
        with open(self.gt_json_path, 'w') as f:
            json.dump(self.records, f, indent=2)

    def _save_variant_records(self):
        cal = self.calibration
        meta = {
            'rig': RIG,
            'psm_name': self.psm_name,
            'primary_variant': PRIMARY_VARIANT,
            'variants': list(VARIANT_COLORS),
            'image_field': 'right_rect_image',
            # stereo-rectified right image: project cam_T_* with P's 3x3 and
            # zero distortion, same as gripper_ground_truth.json
            'camera_matrix_right_rect': self.camera_matrix.tolist(),
            'dist_coeffs_right_rect': [0.0] * 5,
            'rect_T_cam_raw': self.rect_T_cam_raw.tolist(),
            'stereo_rectification': {
                'calib_dir': self.rectification['calib_dir'],
                'left': stereo_side_record(self.rectification['sides']['left']),
                'right': stereo_side_record(self.rectification['sides']['right']),
            },
            # what the rig published at capture (for reference only)
            'right_camera_info': (camera_info_record(self.latest_camera_info['right'])
                                  if 'right' in self.latest_camera_info else None),
            'left_camera_info': (camera_info_record(self.latest_camera_info['left'])
                                 if 'left' in self.latest_camera_info else None),
            'live_K_matches_stereo_K': self._live_K_matches_stereo_K(),
            'hand_eye_json': os.path.abspath(self.hand_eye_json),
            'cam_T_base_post_alignment': cal['post'].tolist(),
            'cam_T_base_pre_alignment': cal['pre'].tolist(),
            'pre_alignment_source': cal['pre_source'],
            'alignment_applied': cal['alignment_applied'],
            'alignment_history': cal['alignment_history'],
            'cam_T_ecm_tip': CAM_T_ECM_TIP.tolist(),
            'pose_topics': self.pose_topics,
        }
        with open(self.variants_json_path, 'w') as f:
            json.dump({'meta': meta, 'records': self.variant_records}, f, indent=2)

    def _live_K_matches_stereo_K(self):
        # The hand-eye calibration was solved against whatever camera_info the
        # rig published then; if that differs from the stereo calibration
        # used here, cam_T_base's frame is only approximately the one these
        # rectification maps assume -- worth redoing the hand-eye calibration.
        info = self.latest_camera_info.get('right')
        if info is None:
            return None
        live_K = numpy.array(info.k).reshape(3, 3)
        return bool(numpy.allclose(live_K, self.rectification['sides']['right']['K'], atol=1.0))

    def _variant_record(self, cam_raw_T_gripper, cam_raw_T_shaft_end, **extra):
        cam_T_gripper = self.rect_T_cam_raw @ cam_raw_T_gripper
        cam_T_shaft_end = self.rect_T_cam_raw @ cam_raw_T_shaft_end
        return {
            **extra,
            'cam_T_gripper': cam_T_gripper.tolist(),
            'cam_T_shaft_end': cam_T_shaft_end.tolist(),
            'cam_raw_T_gripper': cam_raw_T_gripper.tolist(),
            'cam_raw_T_shaft_end': cam_raw_T_shaft_end.tolist(),
            'gripper_pixel_right_rect': project_origin(self.camera_matrix, cam_T_gripper),
            'shaft_end_pixel_right_rect': project_origin(self.camera_matrix, cam_T_shaft_end),
        }

    def callback(self, left_msg, right_msg):
        # Keep this cheap -- runs on crtk's single-threaded executor. Capture
        # (which needs a blocking forward_kinematics call, only safe from the
        # main thread) happens in try_capture(), called from main()'s loop.
        self._pending = (left_msg, right_msg)

    def automatic_capture_due(self) -> bool:
        return self.automatic and time.time() - self.last_save >= self.interval

    def update_live(self):
        # Assumes the caller has already created LIVE_WINDOW_NAME.
        if self._pending is None:
            return
        img = rectify(self.rectification, 'right',
                      self.bridge.imgmsg_to_cv2(self._pending[1], desired_encoding='bgr8'))

        now = time.time()
        mode = f'automatic every {self.interval:.1f}s' if self.automatic else '[c]apture'
        lines = [f'Captured {self.count}  --  {mode}  [q/ESC]quit']
        if now - self.start_time < self.startup_delay:
            lines.append(f'Waiting for startup delay ({self.startup_delay - (now - self.start_time):.1f}s left)')
        elif self._missing_or_stale(now):
            lines.append(f'Not ready: missing/stale {", ".join(self._missing_or_stale(now))}')
        else:
            lines.append('Ready to capture')

        for i, line in enumerate(lines):
            y = 25 + i * 25
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(LIVE_WINDOW_NAME, img)

    def try_capture(self) -> bool:
        if self._pending is None:
            print('No synced frame available yet.')
            return False
        left_msg, right_msg = self._pending
        now = time.time()
        if now - self.start_time < self.startup_delay:
            if not self.automatic:
                print('Still within startup delay.')
            return False
        missing = self._missing_or_stale(now)
        if missing:
            print(f'Skipping frame, missing/stale setup-joint/ECM data: {", ".join(missing)}')
            return False
        suj_poses = {name: self.latest_pose[name] for name in self.pose_topics}

        # Read the arm pose before doing any file I/O: this callback runs on
        # crtk's single-threaded executor, so time spent here delays the
        # executor from servicing the measured_cp subscription, which can
        # make crtk consider the state stale and raise TimeoutError.
        try:
            m_cp, _ = self.arm_handle.local.measured_cp()
            psm_cp, _ = self.arm_handle.measured_cp()  # ECM tip -> PSM tip, through the SUJ
            m_jp, _ = self.arm_handle.measured_jp()
            shaft_end_cp = self.arm_handle.local.forward_kinematics(m_jp[0:4])
        except TimeoutError as e:
            print(f'Skipping frame, gripper pose not available: {e}')
            return False

        self.last_save = now

        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        left_rect_img = rectify(self.rectification, 'left', left_img)
        right_rect_img = rectify(self.rectification, 'right', right_img)

        lfname = f'left-{self.count:04d}.png'
        rfname = f'right-{self.count:04d}.png'
        left_rect_fname = f'left_rect-{self.count:04d}.png'
        rect_fname = f'right_rect-{self.count:04d}.png'
        overlay_fname = f'right_rect_overlay-{self.count:04d}.png'
        cv2.imwrite(os.path.join(self.out_dir, lfname), left_img)
        cv2.imwrite(os.path.join(self.out_dir, rfname), right_img)
        cv2.imwrite(os.path.join(self.out_dir, left_rect_fname), left_rect_img)
        cv2.imwrite(os.path.join(self.out_dir, rect_fname), right_rect_img)

        base_T_gripper = measured_cp_to_matrix(m_cp)
        cam_raw_T_gripper = self.cam_T_base @ base_T_gripper
        base_T_shaft_end = measured_cp_to_matrix(shaft_end_cp)
        cam_raw_T_shaft_end = self.cam_T_base @ base_T_shaft_end
        # into the stereo-rectified right frame right_rect-XXXX.png shows
        cam_T_gripper = self.rect_T_cam_raw @ cam_raw_T_gripper
        cam_T_shaft_end = self.rect_T_cam_raw @ cam_raw_T_shaft_end

        overlay_img, shaft_pixel = draw_axis(
            right_rect_img.copy(), self.camera_matrix, self.dist_coeffs, cam_T_shaft_end
        )
        overlay_img = cv2.circle(overlay_img, shaft_pixel, 5, (255, 255, 0), -1)
        overlay_img, gripper_pixel = draw_axis(
            overlay_img, self.camera_matrix, self.dist_coeffs, cam_T_gripper
        )
        overlay_img = cv2.circle(overlay_img, gripper_pixel, 6, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(self.out_dir, overlay_fname), overlay_img)

        # -- calibration_variants.json -------------------------------------
        raw_poses = {name: pose_record(msg, recv_time, now) for name, (msg, recv_time) in suj_poses.items()}
        raw_poses['psm_measured_cp'] = kdl_pose_record(psm_cp, now)
        raw_poses['psm_local_measured_cp'] = kdl_pose_record(m_cp, now)
        raw_poses['psm_shaft_end_local_cp'] = kdl_pose_record(shaft_end_cp, now)

        # the screenshot's chain: World -> ECM tip and World -> PSM RCM, so
        # ECM tip -> PSM RCM -> PSM tip / shaft end
        world_T_ecm_tip = numpy.array(raw_poses['ecm_measured_cp']['matrix'])
        world_T_psm_rcm = numpy.array(raw_poses['suj_psm_local_measured_cp']['matrix'])
        ecm_tip_T_psm_rcm = numpy.linalg.inv(world_T_ecm_tip) @ world_T_psm_rcm
        ecm_tip_T_gripper = ecm_tip_T_psm_rcm @ base_T_gripper
        ecm_tip_T_shaft_end = ecm_tip_T_psm_rcm @ base_T_shaft_end
        # sanity check: the console's own composition (/PSMx/measured_cp)
        # should match ours unless the topics were sampled at different times
        chain_closure_mm = float(numpy.linalg.norm(
            ecm_tip_T_gripper[0:3, 3] - measured_cp_to_matrix(psm_cp)[0:3, 3]) * 1000.0)

        cam_T_base_pre = self.calibration['pre']
        variants = {
            'kinematic': self._variant_record(
                CAM_T_ECM_TIP @ ecm_tip_T_gripper, CAM_T_ECM_TIP @ ecm_tip_T_shaft_end,
                cam_T_ecm_tip=CAM_T_ECM_TIP.tolist()),
            'hand_eye_pre_alignment': self._variant_record(
                cam_T_base_pre @ base_T_gripper, cam_T_base_pre @ base_T_shaft_end,
                cam_T_base=cam_T_base_pre.tolist()),
            'hand_eye_post_alignment': self._variant_record(
                cam_raw_T_gripper, cam_raw_T_shaft_end, cam_T_base=self.cam_T_base.tolist()),
        }

        variants_overlay_fname = f'right_rect_variants_overlay-{self.count:04d}.png'
        variants_img = right_rect_img.copy()
        for i, (name, v) in enumerate(variants.items()):
            color = VARIANT_COLORS[name]
            g, s = v['gripper_pixel_right_rect'], v['shaft_end_pixel_right_rect']
            # skip points behind the camera or wildly off-image (cv2 needs int32)
            on_screen = [p for p in (s, g) if p is not None and max(abs(p[0]), abs(p[1])) < 1e5]
            pts = [tuple(int(round(c)) for c in p) for p in on_screen]
            if len(pts) == 2:
                variants_img = cv2.line(variants_img, pts[0], pts[1], color, 2, cv2.LINE_AA)
            for p in pts:
                variants_img = cv2.circle(variants_img, p, 5, color, -1)
            y = 25 + i * 25
            cv2.putText(variants_img, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(variants_img, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.out_dir, variants_overlay_fname), variants_img)

        self.variant_records.append({
            'frame_index': self.count,
            'timestamp': now,
            'left_image': lfname,
            'right_image': rfname,
            'left_rect_image': left_rect_fname,
            'right_rect_image': rect_fname,
            'variants_overlay_image': variants_overlay_fname,
            'joint_positions': [float(v) for v in m_jp],
            'kinematic_chain': {
                'world_T_ecm_rcm': raw_poses['suj_ecm_local_measured_cp']['matrix'],
                'ecm_rcm_T_ecm_tip': raw_poses['ecm_local_measured_cp']['matrix'],
                'world_T_ecm_tip': world_T_ecm_tip.tolist(),
                'world_T_psm_rcm': world_T_psm_rcm.tolist(),
                'ecm_tip_T_psm_rcm': ecm_tip_T_psm_rcm.tolist(),
                'psm_rcm_T_gripper': base_T_gripper.tolist(),
                'psm_rcm_T_shaft_end': base_T_shaft_end.tolist(),
                'ecm_tip_T_gripper': ecm_tip_T_gripper.tolist(),
                'ecm_tip_T_shaft_end': ecm_tip_T_shaft_end.tolist(),
                'chain_closure_error_mm': chain_closure_mm,
            },
            'raw_poses': raw_poses,
            'variants': variants,
        })
        self._save_variant_records()

        self.records.append({
            'frame_index': self.count,
            'timestamp': now,
            'left_image': lfname,
            'right_image': rfname,
            'left_rect_image': left_rect_fname,
            'right_rect_image': rect_fname,
            'right_rect_overlay_image': overlay_fname,
            'base_T_gripper': base_T_gripper.tolist(),
            'cam_T_gripper': cam_T_gripper.tolist(),
            'base_T_shaft_end': base_T_shaft_end.tolist(),
            'cam_T_shaft_end': cam_T_shaft_end.tolist(),
            'cam_raw_T_gripper': cam_raw_T_gripper.tolist(),
            'cam_raw_T_shaft_end': cam_raw_T_shaft_end.tolist(),
            'rect_T_cam_raw': self.rect_T_cam_raw.tolist(),
            # intrinsics to render cam_T_* onto right_rect_image (zero distortion)
            'camera_matrix_right_rect': self.camera_matrix.tolist(),
            'image_size_right_rect': list(self.rectification['sides']['right']['size']),
            'joint_positions': [float(v) for v in m_jp],
            'gripper_pixel_right_rect': list(gripper_pixel),
            'shaft_end_pixel_right_rect': list(shaft_pixel),
        })
        self._save_records()

        if self._live_K_matches_stereo_K() is False:
            print(f'WARNING: live /{RIG}/right/camera_info K differs from {self.rectification["calib_dir"]} -- '
                  f'the hand-eye calibration may have been solved with other intrinsics.')
        print(f'Saved frame {self.count}: {lfname}, {rfname}, {left_rect_fname}, {rect_fname} '
              f'(gripper px {gripper_pixel})')
        self.count += 1
        return True


def main():
    argv = crtk.ral.parse_argv(sys.argv[1:])  # skip argv[0], script name

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--psm-name', type=str, default='PSM1', choices=['PSM1', 'PSM2', 'PSM3'],
        help='PSM name corresponding to ROS topics without namespace. Use __ns:= to specify the namespace '
             '(default: PSM1)',
    )
    parser.add_argument(
        '-H', '--hand-eye-json', type=str, default=DEFAULT_HAND_EYE_JSON,
        help='hand-eye calibration matrix in JSON format using OpenCV coordinate system '
             f'(default: {DEFAULT_HAND_EYE_JSON}, where camera_registration.py always writes it)',
    )
    parser.add_argument(
        '--pre-alignment-json', type=str, default=None,
        help='calibration file whose base-frame.transform is the raw (pre-GUI-alignment) hand-eye result, '
             'e.g. one of --hand-eye-json\'s .bak-* backups (default: --hand-eye-json\'s own pre-alignment '
             'block, written by hand_eye_rodrigues_dataset.py when it applies a correction)',
    )
    parser.add_argument(
        '-S', '--stereo-calib-dir', type=str, default=DEFAULT_STEREO_CALIB_DIR,
        help='output_path of stereo_camera_calibration_archits_charuco_board.sh, holding the left.yaml/right.yaml '
             f'used to stereo-rectify the raw images (default: {DEFAULT_STEREO_CALIB_DIR})',
    )
    parser.add_argument(
        '--max-pose-age', type=float, default=1.0,
        help='maximum age in seconds of a cached setup-joint/ECM pose to still be considered valid',
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default='gt_hand_eye_dataset',
        help='directory to save image pairs and gripper_ground_truth.json in',
    )
    parser.add_argument(
        '--automatic', action='store_true',
        help='capture a frame automatically every --interval seconds instead of on a \'c\' keypress',
    )
    parser.add_argument(
        '--interval', type=float, default=2.0,
        help='with --automatic, minimum time in seconds between captures',
    )
    parser.add_argument(
        '--startup-delay', type=float, default=5.0,
        help='seconds to wait after connecting before saving the first frame',
    )
    args = parser.parse_args(argv)

    ral = crtk.ral('image_generation_gt_hand_eye')
    # expected_interval is crtk's staleness timeout for measured_cp/measured_js.
    # The default (0.1s) is tuned for tight control loops; it's too strict here
    # since this node's single-threaded executor also does image file I/O.
    arm_handle = dvrk_camera_registration.ARM(ral, arm_name=args.psm_name, expected_interval=0.5)
    ral.spin()
    ral.check_connections()

    calibration = load_calibration_variants(args.hand_eye_json, args.pre_alignment_json)
    pre_post_mm = numpy.linalg.norm(calibration['pre'][0:3, 3] - calibration['post'][0:3, 3]) * 1000.0
    print(f'Hand-eye: post-alignment from {args.hand_eye_json} (primary); pre-alignment from '
          f'{calibration["pre_source"] or "(same as post)"} ({pre_post_mm:.2f}mm apart)')

    rectification = load_stereo_rectification(args.stereo_calib_dir)

    capture = GroundTruthCapture(ral, arm_handle, calibration, rectification, args.output_dir, args.interval,
                                 args.startup_delay, automatic=args.automatic, psm_name=args.psm_name,
                                 hand_eye_json=args.hand_eye_json, max_pose_age=args.max_pose_age)

    print(f'Waiting for synced image pairs... (capture starts in {args.startup_delay:.0f}s)')
    if args.automatic:
        print(f'Automatic mode: capturing every {args.interval:.1f}s. Press q or ESC to quit.')
    else:
        print('Press c to capture a frame, q or ESC to quit.')

    cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while not ral.is_shutdown():
            capture.update_live()
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('c') and not args.automatic:
                capture.try_capture()
            elif capture.automatic_capture_due():
                capture.try_capture()
    except KeyboardInterrupt:
        pass
    cv2.destroyWindow(LIVE_WINDOW_NAME)

    print(f'Captured {capture.count} frame(s) in ./{args.output_dir}/')
    ral.shutdown()


if __name__ == '__main__':
    main()
