#!/usr/bin/env python3

# Live version of hand_eye_visualization.py's CAD overlay: instead of
# scrubbing through a pre-recorded hand_eye_gt_raw.py dataset, this connects
# to the running dVRK + camera feed directly (same topics/PSM1 handle as
# hand_eye_gt_raw.py) and, for every synced rectified-right frame + PSM1
# pose, projects the tool joint CAD model (anchored to the shaft-end and
# gripper points via the existing hand-eye calibration, exactly like
# hand_eye_visualization.py/3d_pose_alignment.py) onto the frame. Move the
# gripper around in front of the camera and the overlay tracks it live; the
# overlaid frames are shown on screen AND written out to a video file, so
# you end up with a recording of the live feed with the CAD model overlay
# baked in for every frame.

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

WINDOW_NAME = 'Hand-eye CAD overlay (live)'

# camera_registration.py always writes PSM1's hand-eye calibration here
DEFAULT_HAND_EYE_JSON = os.path.expanduser('~/PSM1-registration-open-cv.json')

# the SurgRIPE sample dataset's joint model, matching hand_eye_visualization.py's default
DEFAULT_JOINT_MODEL = os.path.expanduser('~/TEST/joint.npy')


def load_hand_eye_calibration(json_file: str) -> numpy.ndarray:
    with open(json_file, 'r') as f:
        data = json.load(f)
    return numpy.array(data['base-frame']['transform']).reshape(4, 4)


def kdl_frame_to_matrix(frame) -> numpy.ndarray:
    # frame: PyKDL.Frame, as returned by crtk's measured_cp()/forward_kinematics()
    matrix = numpy.eye(4)
    matrix[0:3, 0:3] = numpy.float64(Rotation.from_quat(frame.M.GetQuaternion()).as_matrix())
    matrix[0:3, 3] = (frame.p[0], frame.p[1], frame.p[2])
    return matrix


def draw_axis(img, camera_matrix, dist_coeffs, pose, size=0.01):
    # Mirrors hand_eye_gt_raw.py / hand_eye_visualization.py's draw_axis
    # (itself mirroring dvrk_camera_registration's vis_gripper_pose.py).
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
    points_m = numpy.asarray(points_mm, dtype=numpy.float64) / 1000.0
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


class LiveOverlayRecorder:
    def __init__(self, node, arm_handle, cam_T_base: numpy.ndarray, joint_points_mm: numpy.ndarray,
                 joint_dot_px: int, marker_size: float,
                 out_video: str, fps: float, max_pose_age: float, show: bool):
        self.node = node
        self.arm_handle = arm_handle
        self.cam_T_base = cam_T_base
        self.joint_points_mm = joint_points_mm
        self.joint_dot_px = joint_dot_px
        self.marker_size = marker_size
        self.out_video = out_video
        self.fps = fps
        self.max_pose_age = max_pose_age
        self.show = show

        self.bridge = cv_bridge.CvBridge()
        self.dist_coeffs = numpy.zeros((5, 1))
        self.camera_matrix = None  # derived from right (rectified) camera_info's P
        self.writer = None
        self.frame_count = 0
        self.latest_camera_info_time = None

        self._pending = None

        right_rect_topic = f'/{RIG}/right/image_rect'
        right_info_topic = f'/{RIG}/right/camera_info'

        print(f'Subscribing to {right_rect_topic}, {right_info_topic}')
        print(f'Using PSM1 crtk handle for gripper pose and shaft-end forward kinematics')
        print(f'Recording overlay video to {self.out_video}')

        self.node.create_subscription(
            sensor_msgs.msg.CameraInfo, right_info_topic, self._info_callback, 10
        )
        right_rect_sub = message_filters.Subscriber(self.node, sensor_msgs.msg.Image, right_rect_topic)
        self.ts = message_filters.ApproximateTimeSynchronizer([right_rect_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self._image_callback)

        if self.show:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def _info_callback(self, info_msg):
        projection_matrix = numpy.array(info_msg.p).reshape((3, 4))
        self.camera_matrix = projection_matrix[0:3, 0:3]
        self.latest_camera_info_time = time.time()

    def _image_callback(self, right_rect_msg):
        # Keep this callback cheap: it runs on crtk's single-threaded
        # executor, and process_pending() below makes a blocking
        # forward_kinematics service call, which is only safe from the main
        # thread (see process_pending()), same reasoning as hand_eye_gt_raw.py.
        self._pending = right_rect_msg

    def process_pending(self):
        if self._pending is None:
            return
        right_rect_msg = self._pending
        self._pending = None

        if self.camera_matrix is None:
            print('Skipping frame, no camera_info received yet for right/camera_info')
            return
        now = time.time()
        if self.latest_camera_info_time is not None and now - self.latest_camera_info_time > self.max_pose_age:
            print('Skipping frame, camera_info is stale')
            return

        try:
            psm1_local_cp, _ = self.arm_handle.local.measured_cp()
            psm1_jp, _ = self.arm_handle.measured_jp()
            shaft_end_local_cp = self.arm_handle.local.forward_kinematics(psm1_jp[0:4])
        except TimeoutError as e:
            print(f'Skipping frame, PSM1 pose/kinematics not available: {e}')
            return

        img = self.bridge.imgmsg_to_cv2(right_rect_msg, desired_encoding='bgr8')

        base_T_gripper = kdl_frame_to_matrix(psm1_local_cp)
        base_T_shaft_end = kdl_frame_to_matrix(shaft_end_local_cp)
        cam_T_gripper = self.cam_T_base @ base_T_gripper
        cam_T_shaft = self.cam_T_base @ base_T_shaft_end

        anchor = cam_T_joint_model(cam_T_shaft, cam_T_gripper)
        img = draw_joint_model(img, self.camera_matrix, self.dist_coeffs, anchor,
                                self.joint_points_mm, dot_px=self.joint_dot_px)
        img, shaft_origin = draw_axis(img, self.camera_matrix, self.dist_coeffs, cam_T_shaft, self.marker_size)
        img = cv2.circle(img, shaft_origin, 5, (255, 255, 0), -1)
        img, gripper_origin = draw_axis(img, self.camera_matrix, self.dist_coeffs, cam_T_gripper, self.marker_size)
        img = cv2.circle(img, gripper_origin, 6, (0, 0, 255), -1)

        line = f'frame {self.frame_count}  (hand-eye CAD overlay, live)'
        cv2.putText(img, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if self.writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.out_video, fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f'Could not open {self.out_video} for writing -- check the extension/codec')

        self.writer.write(img)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            print(f'Recorded {self.frame_count} frames...')

        if self.show:
            cv2.imshow(WINDOW_NAME, img)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f'Saved {self.frame_count} frames to {self.out_video}')
        if self.show:
            cv2.destroyAllWindows()


def main():
    argv = crtk.ral.parse_argv(sys.argv[1:])  # also calls rclpy.init(), skip argv[0]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-H', '--hand-eye-json', type=str, default=DEFAULT_HAND_EYE_JSON,
        help='hand-eye calibration matrix in JSON format using OpenCV coordinate system '
             f'(default: {DEFAULT_HAND_EYE_JSON}, where camera_registration.py always writes it)',
    )
    parser.add_argument(
        '-o', '--output', type=str, default='hand_eye_cad_overlay.mp4',
        help='output video file path (default: hand_eye_cad_overlay.mp4)',
    )
    parser.add_argument(
        '--fps', type=float, default=15.0,
        help='frame rate of the recorded video (default: 15)',
    )
    parser.add_argument(
        '--joint-model', type=str, default=DEFAULT_JOINT_MODEL,
        help='path to a .npy Nx3 point cloud of the tool joint (CAD model), e.g. SurgRIPE\'s joint.npy '
             f'(default: {DEFAULT_JOINT_MODEL})',
    )
    parser.add_argument(
        '--joint-model-dot-px', type=int, default=3,
        help='pixel size of each drawn joint-model point after dilation (default: 3)',
    )
    parser.add_argument(
        '--marker-size', type=float, default=0.01,
        help='axis marker length in meters (default: 0.01)',
    )
    parser.add_argument(
        '--max-pose-age', type=float, default=1.0,
        help='maximum age in seconds of the cached camera_info message to still be considered valid',
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='do not open a live preview window, just record the video (useful when headless)',
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.joint_model):
        parser.error(f'{args.joint_model} not found -- pass --joint-model to point at a joint .npy point cloud')

    cam_T_base = load_hand_eye_calibration(args.hand_eye_json)
    joint_points_mm = numpy.load(args.joint_model)

    ral = crtk.ral('hand_eye_video_cad_overlay')
    # expected_interval is crtk's staleness timeout for measured_cp/measured_js.
    # The default (0.1s) is tuned for tight control loops; it's too strict here
    # since this node's executor also does image display/encoding.
    arm_handle = dvrk_camera_registration.ARM(ral, arm_name='PSM1', expected_interval=0.5)
    ral.spin()
    ral.check_connections()

    recorder = LiveOverlayRecorder(
        ral._node, arm_handle, cam_T_base, joint_points_mm, args.joint_model_dot_px,
        args.marker_size, args.output, args.fps, args.max_pose_age, show=not args.no_display,
    )

    print('Move the gripper in view of the camera -- recording starts as soon as frames/poses are available.')
    print("Press 'q' or Ctrl+C to stop and save the video." if not args.no_display else 'Press Ctrl+C to stop and save the video.')
    try:
        while not ral.is_shutdown():
            recorder.process_pending()
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):  # q or ESC
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    recorder.close()
    ral.shutdown()


if __name__ == '__main__':
    main()
