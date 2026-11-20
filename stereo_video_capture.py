#!/usr/bin/env python3

# Stereo frames straight from the DeckLink cards through the
# DecklinkVideoGstreamerClient python bindings (stereo_video module),
# replacing the dvrk_video/gscam ROS image topics for every script that
# records images.
#
# A video_source must be running (it owns the DeckLink cards -- gscam /
# dvrk_video's decklink launch files can NOT run at the same time, the
# cards are exclusive). Start it with ./start_video_source.sh. Any number
# of clients (these scripts, stereo_video_ros_bridge.py, stereo_viewer,
# stereo_recorder) can then connect at once.
#
# What the old ROS path gave and how it maps here:
#   /RIG/{left,right}/image_raw    -> StereoFrame.left / .right (bgr8)
#   /RIG/{left,right}/camera_info  -> load_camera_info(side), read from the
#                                     same dvrk_video calibrations/RIG/*.yaml
#                                     gscam loads (camera_info_url)
#   /RIG/{left,right}/image_rect   -> CameraInfoRectifier(info)(image), the
#                                     same maps image_proc's rectify node
#                                     builds (bit-identical output)
# left and right come from one composited frame, so they're always a pair
# (no ApproximateTimeSynchronizer needed).

import os
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy
import yaml

RIG = os.environ.get('RIG', 'jhu_dVRK')  # falls back if $RIG isn't set

DEFAULT_SOURCE_NAME = os.environ.get('STEREO_VIDEO_SOURCE', 'camera')
# where `cmake -B build -DBUILD_PYTHON=ON` puts stereo_video.cpython-*.so
DEFAULT_CLIENT_BUILD_DIR = os.environ.get(
    'STEREO_VIDEO_BUILD', os.path.expanduser('~/DecklinkVideoGstreamerClient/build'))

# A frame older than this is treated as no frame (source stopped / stalled),
# so nothing stale ever gets recorded.
DEFAULT_MAX_FRAME_AGE = 0.5


def import_stereo_video():
    try:
        import stereo_video
        return stereo_video
    except ImportError:
        pass
    if os.path.isdir(DEFAULT_CLIENT_BUILD_DIR) and DEFAULT_CLIENT_BUILD_DIR not in sys.path:
        sys.path.insert(0, DEFAULT_CLIENT_BUILD_DIR)
    try:
        import stereo_video
        return stereo_video
    except ImportError as e:
        raise SystemExit(
            f'Cannot import the stereo_video python bindings ({e}). Build them with\n'
            f'  cd ~/DecklinkVideoGstreamerClient && cmake -B build -DBUILD_PYTHON=ON '
            f'-DPython_EXECUTABLE=/usr/bin/python3 && cmake --build build -j\n'
            f'or set STEREO_VIDEO_BUILD to the directory holding stereo_video.cpython-*.so')


# ---------------------------------------------------------------------------
# camera_info (what gscam used to publish)
# ---------------------------------------------------------------------------

def calibration_yaml_path(side: str, rig: str = RIG) -> str:
    # gscam's camera_info_url is package://dvrk_video/calibrations/<rig>/<side>.yaml
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('dvrk_video')
    except Exception:
        share = os.path.expanduser('~/ros2_ws/src/dvrk/dvrk_video')
    return os.path.join(share, 'calibrations', rig, f'{side}.yaml')


def load_camera_info(side: str, rig: str = RIG, path: str = None):
    # -> sensor_msgs.msg.CameraInfo, filled exactly like gscam's
    # camera_info_manager does (frame_id included), so camera_info_record()
    # and friends keep working unchanged.
    import sensor_msgs.msg

    path = path or calibration_yaml_path(side, rig)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    msg = sensor_msgs.msg.CameraInfo()
    msg.header.frame_id = f'/{rig}/{side}_frame'  # gscam: "/" + camera_name + "_frame"
    msg.width = int(data['image_width'])
    msg.height = int(data['image_height'])
    msg.distortion_model = data.get('distortion_model', 'plumb_bob')
    msg.d = [float(v) for v in data['distortion_coefficients']['data']]
    msg.k = [float(v) for v in data['camera_matrix']['data']]
    msg.r = [float(v) for v in data['rectification_matrix']['data']]
    msg.p = [float(v) for v in data['projection_matrix']['data']]
    return msg


class CameraInfoRectifier:
    # Same as image_proc's rectify node (image_geometry rectifyImage):
    # initUndistortRectifyMap(K, D, R, P[:3,:3], CV_16SC2) + INTER_LINEAR.
    # Verified bit-identical against recorded /RIG/right/image_rect frames.
    def __init__(self, info):
        K = numpy.array(info.k, dtype=numpy.float64).reshape(3, 3)
        D = numpy.array(info.d, dtype=numpy.float64)
        R = numpy.array(info.r, dtype=numpy.float64).reshape(3, 3)
        P = numpy.array(info.p, dtype=numpy.float64).reshape(3, 4)
        self.size = (int(info.width), int(info.height))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, R, P[0:3, 0:3], self.size, cv2.CV_16SC2)

    def __call__(self, img: numpy.ndarray) -> numpy.ndarray:
        if (img.shape[1], img.shape[0]) != self.size:
            raise ValueError(f'image is {img.shape[1]}x{img.shape[0]} but camera_info is for '
                             f'{self.size[0]}x{self.size[1]}')
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

@dataclass
class StereoFrame:
    seq: int          # increments by one per grabbed frame
    stamp: float      # time.time() when grabbed
    left: numpy.ndarray   # BGR, owned (safe to keep)
    right: numpy.ndarray  # BGR, owned (safe to keep)

    def age(self) -> float:
        return time.time() - self.stamp


class StereoVideoCapture:
    # Grabs continuously on a background thread (like a ROS subscription
    # callback did) and keeps only the latest pair. Reconnects on its own if
    # the video_source is restarted.
    def __init__(self, source_name: str = DEFAULT_SOURCE_NAME, connect_timeout: float = 5.0):
        self._sv = import_stereo_video()
        self.source_name = source_name
        self.socket_path = self._sv.socket_path_for(source_name)
        self._client = self._sv.StereoVideoClient()
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._stop = threading.Event()

        deadline = time.time() + connect_timeout
        while not self._client.init(source_name):
            if time.time() > deadline:
                raise SystemExit(
                    f'Cannot connect to stereo video source "{source_name}" ({self.socket_path}). '
                    f'Start it first: ./start_video_source.sh (and make sure the dvrk_video '
                    f'gscam launch is NOT running -- it holds the DeckLink cards).')
            time.sleep(0.2)

        self._thread = threading.Thread(target=self._run, name='stereo_video_grab', daemon=True)
        self._thread.start()
        print(f'Connected to stereo video source "{source_name}" ({self.socket_path})')

    def _run(self):
        warned = False
        while not self._stop.is_set():
            if not self._client.is_connected():
                if not warned:
                    print(f'WARNING: lost stereo video source "{self.source_name}", reconnecting...')
                    warned = True
                if not self._client.init(self.source_name):
                    time.sleep(0.5)
                    continue
                print(f'Reconnected to stereo video source "{self.source_name}"')
                warned = False
            if not self._client.grab(100):  # releases the GIL while waiting
                continue
            frame = StereoFrame(self._seq, time.time(),
                                self._client.left_image().copy(), self._client.right_image().copy())
            with self._lock:
                self._latest = frame
                self._seq += 1

    def latest(self, max_age: float = DEFAULT_MAX_FRAME_AGE):
        # latest StereoFrame, or None if none yet / older than max_age
        with self._lock:
            frame = self._latest
        if frame is None or (max_age is not None and frame.age() > max_age):
            return None
        return frame

    def wait_for_frame(self, timeout: float = 5.0, max_age: float = DEFAULT_MAX_FRAME_AGE):
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.latest(max_age)
            if frame is not None:
                return frame
            time.sleep(0.01)
        return None

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
