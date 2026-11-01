#!/usr/bin/env python3

# Read-only viewer for hand_eye_gt_raw.py output datasets: scrubs through a
# dataset's frames showing the hand-eye projection as an overlaid CAD model
# (the same real joint-geometry point-cloud render used by
# 3d_pose_alignment.py, placed on the wrist clevis from the shaft-end and
# gripper axes together -- see cam_T_joint_model). If a frame has a saved correction from
# 3d_pose_alignment.py (hand_eye_gt_raw_corrections.json), that correction
# is applied so you're seeing the aligned result, not the raw nominal
# projection. No editing here -- just a quick way to review a dataset's
# hand-eye accuracy, e.g. to check a freshly re-collected "after" dataset
# against a calibration update.

import argparse
import json
import os
import sys

import cv2
import numpy

WINDOW_NAME = 'Hand-eye CAD overlay'

GT_JSON_NAME = 'hand_eye_gt_raw.json'
CORRECTIONS_JSON_NAME = 'hand_eye_gt_raw_corrections.json'

# the SurgRIPE sample dataset's joint model -- the one this project has
# always used, fixed rather than a flag since it's never actually changed
DEFAULT_JOINT_MODEL = os.path.expanduser('~/TEST/joint.npy')


def _project(points_3d, camera_matrix, dist_coeffs) -> numpy.ndarray:
    # points_3d are already expressed in the camera frame, so project with
    # zero rvec/tvec rather than re-applying a pose.
    zero = numpy.zeros(3)
    points_2d, _ = cv2.projectPoints(numpy.asarray(points_3d, dtype=numpy.float32),
                                      zero, zero, camera_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2).astype(numpy.int32)


def draw_axis(img, camera_matrix, dist_coeffs, pose, size=0.01):
    # Mirrors hand_eye_gt_raw.py's draw_axis (itself mirroring
    # dvrk_camera_registration's vis_gripper_pose.py).
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


class Dataset:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(os.path.normpath(path))

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

        corrections_path = os.path.join(path, CORRECTIONS_JSON_NAME)
        self.corrections = {}
        if os.path.exists(corrections_path):
            with open(corrections_path, 'r') as f:
                self.corrections = json.load(f).get('corrections', {})

    def image_path(self, record) -> str:
        return os.path.join(self.path, record['right_rect_image'])


class Viewer:
    def __init__(self, datasets, joint_points_mm, joint_dot_px, marker_size):
        self.frames = [(ds, i, record) for ds in datasets for i, record in enumerate(ds.records)]
        self.joint_points_mm = joint_points_mm
        self.joint_dot_px = joint_dot_px
        self.marker_size = marker_size

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Image', WINDOW_NAME, 0, len(self.frames) - 1, self._on_trackbar)
        self.index = 0
        self._render()

    def _on_trackbar(self, pos):
        self.index = pos
        self._render()

    def _go(self, delta):
        new_index = max(0, min(len(self.frames) - 1, self.index + delta))
        cv2.setTrackbarPos('Image', WINDOW_NAME, new_index)  # triggers _on_trackbar

    def _render(self):
        ds, frame_index, record = self.frames[self.index]
        img = cv2.imread(ds.image_path(record))
        if img is None:
            print(f'Could not read {ds.image_path(record)}')
            return

        cam_T_gripper = numpy.array(record['cam_T_gripper']).reshape((4, 4))
        has_shaft = 'cam_T_shaft_end' in record
        cam_T_shaft = numpy.array(record['cam_T_shaft_end']).reshape((4, 4)) if has_shaft else None

        correction = ds.corrections.get(str(frame_index))
        if correction is not None:
            delta_T = numpy.array(correction['delta_T']).reshape((4, 4))
            cam_T_gripper = delta_T @ cam_T_gripper
            if has_shaft:
                cam_T_shaft = delta_T @ cam_T_shaft

        if has_shaft:
            anchor = cam_T_joint_model(cam_T_shaft, cam_T_gripper)
            img = draw_joint_model(img, ds.camera_matrix, ds.dist_coeffs, anchor,
                                    self.joint_points_mm, dot_px=self.joint_dot_px)
            img, shaft_origin = draw_axis(img, ds.camera_matrix, ds.dist_coeffs, cam_T_shaft, self.marker_size)
            img = cv2.circle(img, shaft_origin, 5, (255, 255, 0), -1)

        img, gripper_origin = draw_axis(img, ds.camera_matrix, ds.dist_coeffs, cam_T_gripper, self.marker_size)
        img = cv2.circle(img, gripper_origin, 6, (0, 0, 255), -1)

        status = 'corrected' if correction is not None else 'nominal'
        line = f'{ds.name}  frame {frame_index} ({self.index + 1}/{len(self.frames)} total)  [{status}]'
        cv2.putText(img, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        hint = '[n]ext  [p]rev  [q]uit'
        cv2.putText(img, hint, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, hint, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, img)

    def run(self):
        print(f'{len(self.frames)} frames loaded from {len(set(id(ds) for ds, _, _ in self.frames))} dataset(s).')
        print("Use the 'Image' slider or n/p to step through frames, q to quit.")
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('n'):
                self._go(1)
            elif key == ord('p'):
                self._go(-1)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'datasets', type=str, nargs='+',
        help='one or more hand_eye_gt_raw.py output directories to review',
    )
    parser.add_argument(
        '--joint-model-dot-px', type=int, default=3,
        help='pixel size of each drawn joint-model point after dilation (default: 3)',
    )
    parser.add_argument(
        '--marker-size', type=float, default=0.01,
        help='axis marker length in meters (default: 0.01)',
    )
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(DEFAULT_JOINT_MODEL):
        parser.error(f'{DEFAULT_JOINT_MODEL} not found -- this is the fixed joint model path this script uses')

    datasets = [Dataset(path) for path in args.datasets]
    joint_points_mm = numpy.load(DEFAULT_JOINT_MODEL)

    viewer = Viewer(datasets, joint_points_mm, args.joint_model_dot_px, args.marker_size)
    viewer.run()


if __name__ == '__main__':
    main()
