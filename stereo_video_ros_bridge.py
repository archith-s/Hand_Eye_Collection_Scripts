#!/usr/bin/env python3

# Republishes the DecklinkVideoGstreamerClient stereo video source on the
# same ROS topics `ros2 launch dvrk_video decklink_stereo_goovis.launch.py`
# (gscam + image_proc rectify) used to publish, for the tools that still read
# images over ROS -- dvrk_camera_registration's camera_registration.py and
# vis_gripper_pose.py:
#
#   /RIG/{left,right}/image_raw     bgr8
#   /RIG/{left,right}/camera_info   from dvrk_video calibrations/RIG/*.yaml
#   /RIG/{left,right}/image_rect    same maps as image_proc rectify
#
# The recording scripts in this repo read the source directly (see
# stereo_video_capture.py) and don't need this. Run it alongside them freely:
# the source serves any number of clients.
#
#   ./start_video_source.sh                      # terminal 1
#   python3 stereo_video_ros_bridge.py           # terminal 2
#   ros2 run dvrk_camera_registration camera_registration.py -p PSM1 -m 0.015 -c /${RIG}/right -M safe

import argparse

import rclpy
import sensor_msgs.msg
from cv_bridge import CvBridge
from rclpy.node import Node

import stereo_video_capture as svc


class StereoVideoBridge(Node):
    def __init__(self, rig: str, source_name: str, rectify: bool):
        super().__init__('stereo_video_ros_bridge')
        self.bridge = CvBridge()
        self.capture = svc.StereoVideoCapture(source_name)
        self.infos = {}
        self.rectifiers = {}
        self.pubs = {}
        for side in ('left', 'right'):
            path = svc.calibration_yaml_path(side, rig)
            self.infos[side] = svc.load_camera_info(side, rig, path)
            self.rectifiers[side] = svc.CameraInfoRectifier(self.infos[side]) if rectify else None
            ns = f'/{rig}/{side}'
            self.pubs[side] = {
                'raw': self.create_publisher(sensor_msgs.msg.Image, f'{ns}/image_raw', 1),
                'info': self.create_publisher(sensor_msgs.msg.CameraInfo, f'{ns}/camera_info', 1),
                'rect': self.create_publisher(sensor_msgs.msg.Image, f'{ns}/image_rect', 1) if rectify else None,
            }
            self.get_logger().info(f'{ns}: image_raw, camera_info ({path})' + (', image_rect' if rectify else ''))
        self.last_seq = -1
        self.timer = self.create_timer(0.005, self._tick)

    def _tick(self):
        frame = self.capture.latest()
        if frame is None or frame.seq == self.last_seq:
            return
        self.last_seq = frame.seq
        stamp = self.get_clock().now().to_msg()
        for side, img in (('left', frame.left), ('right', frame.right)):
            pubs = self.pubs[side]
            info = self.infos[side]
            info.header.stamp = stamp
            raw = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            raw.header.stamp = stamp
            raw.header.frame_id = info.header.frame_id
            pubs['raw'].publish(raw)
            pubs['info'].publish(info)
            if pubs['rect'] is not None and pubs['rect'].get_subscription_count() > 0:
                rect = self.bridge.cv2_to_imgmsg(self.rectifiers[side](img), encoding='bgr8')
                rect.header = raw.header
                pubs['rect'].publish(rect)

    def destroy_node(self):
        self.capture.close()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rig', default=svc.RIG, help='stereo rig name / topic namespace (default $RIG)')
    parser.add_argument('--source', default=svc.DEFAULT_SOURCE_NAME, help='video_source -n name')
    parser.add_argument('--no-rectify', action='store_true', help="don't publish image_rect")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = StereoVideoBridge(args.rig, args.source, not args.no_rectify)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
