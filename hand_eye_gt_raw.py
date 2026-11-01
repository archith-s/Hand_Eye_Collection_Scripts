#!/usr/bin/env python3

# Captures synced left/right raw image pairs (same approach as
# charuco_image_generation.py / image_generation_gt_hand_eye.py) together
# with the raw poses of PSM1, ECM, and their SUJ (setup joint) counterparts,
# plus left/right camera intrinsics. An existing hand-eye calibration (as
# loaded by dvrk_camera_registration's vis_gripper_pose.py) is used to also
# express two PSM1 reference poses in the camera frame and draw a debug
# overlay: the gripper tip (like image_generation_gt_hand_eye.py) AND the
# end of the shaft (forward kinematics on the first 4 joints, excluding the
# wrist), exactly like vis_gripper_pose.py's own two-axis display -- the
# shaft-end point is a useful second reference since, unlike the tip, it
# doesn't move with wrist/jaw articulation.
#
# Unlike image_generation_gt_hand_eye.py, this script also records the raw
# PSM1/ECM/SUJ poses themselves (not just the poses derived from the
# supplied hand-eye calibration). The intent is to run this script once per
# camera angle/setup-joint position, producing one dataset per angle; a
# separate GUI tool (3d_pose_alignment.py) can then use the raw poses +
# images from multiple angles to do 3D pose alignment and refine the
# hand-eye calibration.

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
from scipy.spatial.transform import Rotation

import crtk
import dvrk_camera_registration

RIG = os.environ.get('RIG', 'jhu_dVRK')  # falls back if $RIG isn't set

# camera_registration.py always writes PSM1's hand-eye calibration here
DEFAULT_HAND_EYE_JSON = os.path.expanduser('~/PSM1-registration-open-cv.json')

# name -> topic, for the raw PoseStamped topics (besides PSM1's, which come
# from the crtk ARM handle below since we also need its forward_kinematics
# service) to record alongside each frame
POSE_TOPICS = {
    'ecm_measured_cp': '/ECM/measured_cp',
    'ecm_local_measured_cp': '/ECM/local/measured_cp',
    'suj_psm1_measured_cp': '/SUJ/PSM1/measured_cp',
    'suj_psm1_local_measured_cp': '/SUJ/PSM1/local/measured_cp',
    'suj_ecm_measured_cp': '/SUJ/ECM/measured_cp',
    'suj_ecm_local_measured_cp': '/SUJ/ECM/local/measured_cp',
}

# the two PSM1 poses used, together with the supplied hand-eye calibration,
# to compute cam_T_gripper / cam_T_shaft_end
GRIPPER_POSE_NAME = 'psm1_local_measured_cp'
SHAFT_POSE_NAME = 'psm1_shaft_end_local_cp'


def load_hand_eye_calibration(json_file: str) -> numpy.ndarray:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return numpy.array(data['base-frame']['transform']).reshape(4, 4)


def pose_stamped_to_matrix(msg: geometry_msgs.msg.PoseStamped) -> numpy.ndarray:
    q = msg.pose.orientation
    p = msg.pose.position
    return quaternion_position_to_matrix((q.x, q.y, q.z, q.w), (p.x, p.y, p.z))


def kdl_frame_to_matrix(frame) -> numpy.ndarray:
    # frame: PyKDL.Frame, as returned by crtk's measured_cp()/forward_kinematics()
    return quaternion_position_to_matrix(frame.M.GetQuaternion(), (frame.p[0], frame.p[1], frame.p[2]))


def quaternion_position_to_matrix(quat_xyzw, position) -> numpy.ndarray:
    matrix = numpy.eye(4)
    matrix[0:3, 0:3] = numpy.float64(Rotation.from_quat(quat_xyzw).as_matrix())
    matrix[0:3, 3] = position
    return matrix


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
    # No ROS header on a PyKDL.Frame -- this is fetched synchronously right
    # now, so stamp/age are just `now`/0 for a consistent record shape.
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


class RawGroundTruthCapture:
    def __init__(self, node, arm_handle, cam_T_base: numpy.ndarray, hand_eye_json: str, out_dir: str,
                 interval: float, startup_delay: float, max_pose_age: float):
        self.node = node
        self.arm_handle = arm_handle
        self.cam_T_base = cam_T_base
        self.out_dir = out_dir
        self.interval = interval
        self.startup_delay = startup_delay
        self.max_pose_age = max_pose_age
        self.start_time = time.time()
        os.makedirs(self.out_dir, exist_ok=True)

        self.gt_json_path = os.path.join(self.out_dir, 'hand_eye_gt_raw.json')
        self.records, meta = self._load_existing()
        self.count = len(self.records)
        self.last_save = 0.0

        self.bridge = cv_bridge.CvBridge()
        self.dist_coeffs = numpy.zeros((5, 1))

        # latest received message + wall-clock receive time, keyed by name
        self.latest_pose = {}
        self.latest_camera_info = {}
        self.camera_matrix = None  # derived from right camera_info's P, used for overlay

        # set by the (lightweight) image sync callback, consumed by
        # process_pending() on the main thread -- see process_pending()'s
        # docstring for why the heavy work happens there instead of here.
        self._pending = None

        left_topic = f'/{RIG}/left/image_raw'
        right_topic = f'/{RIG}/right/image_raw'
        left_info_topic = f'/{RIG}/left/camera_info'
        right_info_topic = f'/{RIG}/right/camera_info'
        # The hand-eye calibration (and vis_gripper_pose.py, which this script's
        # projection math mirrors) is defined relative to the RECTIFIED right
        # camera frame, projected with its P matrix and zero distortion. Raw
        # image_raw frames are still distorted, so we additionally grab the
        # rectified right image just to draw a debug overlay that is
        # correctly aligned; right/camera_info carries the P matrix used for
        # both this overlay and (via K/D) the raw right image's intrinsics.
        right_rect_topic = f'/{RIG}/right/image_rect'

        print(f'Loaded hand-eye calibration from {hand_eye_json}')
        print(f'Subscribing to {left_topic}, {right_topic}, {right_rect_topic}')
        print(f'Subscribing to {left_info_topic}, {right_info_topic}')
        for name, topic in POSE_TOPICS.items():
            print(f'Subscribing to {topic} ({name})')
        print(f'Using PSM1 crtk handle for {GRIPPER_POSE_NAME} and {SHAFT_POSE_NAME} (forward kinematics)')
        print(f'Saving frames + ground truth to ./{self.out_dir}/')

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

    def _load_existing(self):
        if os.path.exists(self.gt_json_path):
            with open(self.gt_json_path, 'r') as f:
                data = json.load(f)
            return data.get('records', []), data.get('meta', {})
        return [], {}

    def _save(self):
        meta = {
            'rig': RIG,
            'cam_T_base': self.cam_T_base.tolist(),
            'gripper_pose_name': GRIPPER_POSE_NAME,
            'shaft_pose_name': SHAFT_POSE_NAME,
            'pose_topics': POSE_TOPICS,
        }
        if 'left' in self.latest_camera_info:
            meta['left_camera_info'] = camera_info_record(self.latest_camera_info['left'][0])
        if 'right' in self.latest_camera_info:
            meta['right_camera_info'] = camera_info_record(self.latest_camera_info['right'][0])
        with open(self.gt_json_path, 'w') as f:
            json.dump({'meta': meta, 'records': self.records}, f, indent=2)

    def _missing_or_stale(self, now):
        missing = []
        for name in POSE_TOPICS:
            entry = self.latest_pose.get(name)
            if entry is None:
                missing.append(f'{name} (not received)')
            elif now - entry[1] > self.max_pose_age:
                missing.append(f'{name} (stale, {now - entry[1]:.2f}s old)')
        for name in ('left', 'right'):
            entry = self.latest_camera_info.get(name)
            if entry is None:
                missing.append(f'{name}_camera_info (not received)')
        return missing

    def _image_callback(self, left_msg, right_msg, right_rect_msg):
        # Keep this callback cheap: it runs on crtk's single-threaded
        # executor, and process_pending() below needs to make a blocking
        # forward_kinematics service call, which is only safe to do from
        # the main thread (see process_pending()).
        self._pending = (left_msg, right_msg, right_rect_msg)

    def process_pending(self):
        # Runs on the main thread (see main()'s loop). PSM1's measured_cp/
        # measured_jp/forward_kinematics calls below block until crtk's
        # background executor thread (started by ral.spin()) services them;
        # calling them from *within* a callback already running on that
        # same executor (as the old single-callback version of this class
        # did) risks reentrant deadlock, so the image sync callback above
        # just stashes the synced triple and this method does the rest.
        if self._pending is None:
            return
        left_msg, right_msg, right_rect_msg = self._pending
        self._pending = None

        now = time.time()
        if now - self.start_time < self.startup_delay:
            return
        if now - self.last_save < self.interval:
            return
        if self.camera_matrix is None:
            print('Skipping frame, no camera_info received yet for right/camera_info')
            return

        missing = self._missing_or_stale(now)
        if missing:
            print(f'Skipping frame, missing/stale data: {", ".join(missing)}')
            return

        try:
            psm1_cp, _ = self.arm_handle.measured_cp()
            psm1_local_cp, _ = self.arm_handle.local.measured_cp()
            psm1_jp, _ = self.arm_handle.measured_jp()
            shaft_end_local_cp = self.arm_handle.local.forward_kinematics(psm1_jp[0:4])
        except TimeoutError as e:
            print(f'Skipping frame, PSM1 pose/kinematics not available: {e}')
            return

        self.last_save = now

        left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='bgr8')
        right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='bgr8')
        right_rect_img = self.bridge.imgmsg_to_cv2(right_rect_msg, desired_encoding='bgr8')

        lfname = f'left-{self.count:04d}.png'
        rfname = f'right-{self.count:04d}.png'
        rect_fname = f'right_rect-{self.count:04d}.png'
        overlay_fname = f'right_rect_overlay-{self.count:04d}.png'
        cv2.imwrite(os.path.join(self.out_dir, lfname), left_img)
        cv2.imwrite(os.path.join(self.out_dir, rfname), right_img)
        cv2.imwrite(os.path.join(self.out_dir, rect_fname), right_rect_img)

        base_T_gripper = kdl_frame_to_matrix(psm1_local_cp)
        base_T_shaft_end = kdl_frame_to_matrix(shaft_end_local_cp)
        cam_T_gripper = self.cam_T_base @ base_T_gripper
        cam_T_shaft_end = self.cam_T_base @ base_T_shaft_end

        overlay_img = right_rect_img.copy()
        overlay_img, shaft_pixel = draw_axis(overlay_img, self.camera_matrix, self.dist_coeffs, cam_T_shaft_end)
        overlay_img, gripper_pixel = draw_axis(overlay_img, self.camera_matrix, self.dist_coeffs, cam_T_gripper)
        overlay_img = cv2.line(overlay_img, shaft_pixel, gripper_pixel, (0, 255, 255), 1)
        overlay_img = cv2.circle(overlay_img, shaft_pixel, 5, (255, 255, 0), -1)
        overlay_img = cv2.circle(overlay_img, gripper_pixel, 6, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(self.out_dir, overlay_fname), overlay_img)

        poses = {
            name: pose_record(msg, recv_time, now)
            for name, (msg, recv_time) in self.latest_pose.items()
        }
        poses['psm1_measured_cp'] = kdl_pose_record(psm1_cp, now)
        poses[GRIPPER_POSE_NAME] = kdl_pose_record(psm1_local_cp, now)
        poses[SHAFT_POSE_NAME] = kdl_pose_record(shaft_end_local_cp, now)

        self.records.append({
            'frame_index': self.count,
            'timestamp': now,
            'left_image': lfname,
            'right_image': rfname,
            'right_rect_image': rect_fname,
            'right_rect_overlay_image': overlay_fname,
            'poses': poses,
            'cam_T_gripper': cam_T_gripper.tolist(),
            'cam_T_shaft_end': cam_T_shaft_end.tolist(),
            'gripper_pixel_right_rect': list(gripper_pixel),
            'shaft_pixel_right_rect': list(shaft_pixel),
        })
        self._save()

        print(f'Saved frame {self.count}: {lfname}, {rfname}, {rect_fname} '
              f'(gripper px {gripper_pixel}, shaft px {shaft_pixel})')
        self.count += 1


def main():
    argv = crtk.ral.parse_argv(sys.argv[1:])  # also calls rclpy.init(), skip argv[0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-H', '--hand-eye-json', type=str, default=DEFAULT_HAND_EYE_JSON,
        help='hand-eye calibration matrix in JSON format using OpenCV coordinate system '
             f'(default: {DEFAULT_HAND_EYE_JSON}, where camera_registration.py always writes it)',
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, default='hand_eye_gt_raw_dataset',
        help='directory to save images and hand_eye_gt_raw.json in (run once per camera angle)',
    )
    parser.add_argument(
        '--interval', type=float, default=2.0,
        help='minimum time in seconds between captures',
    )
    parser.add_argument(
        '--startup-delay', type=float, default=5.0,
        help='seconds to wait after connecting before saving the first frame',
    )
    parser.add_argument(
        '--max-pose-age', type=float, default=1.0,
        help='maximum age in seconds of a cached pose/camera_info message to still be considered valid',
    )
    args = parser.parse_args(argv)

    cam_T_base = load_hand_eye_calibration(args.hand_eye_json)

    ral = crtk.ral('hand_eye_gt_raw')
    # expected_interval is crtk's staleness timeout for measured_cp/measured_js.
    # The default (0.1s) is tuned for tight control loops; it's too strict here
    # since this node's executor also does image file I/O.
    arm_handle = dvrk_camera_registration.ARM(ral, arm_name='PSM1', expected_interval=0.5)
    ral.spin()
    ral.check_connections()

    capture = RawGroundTruthCapture(
        ral._node, arm_handle, cam_T_base, args.hand_eye_json, args.output_dir,
        args.interval, args.startup_delay, args.max_pose_age,
    )

    print(f'Waiting for synced image pairs... (capture starts in {args.startup_delay:.0f}s)')
    try:
        while not ral.is_shutdown():
            capture.process_pending()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

    ral.shutdown()


if __name__ == '__main__':
    main()
