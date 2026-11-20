#!/usr/bin/env python3

# Saves a left/right pair every --interval seconds (default 2) to
# ./charuco_dataset/left-NNNN.png / right-NNNN.png while you move the ChArUco
# board under the endoscope. Frames come straight from the DeckLink cards via
# DecklinkVideoGstreamerClient (start ./start_video_source.sh first) -- no
# ROS needed. Ctrl+C to stop.

import argparse
import os
import time

import cv2

import stereo_video_capture as svc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='charuco_dataset')
    parser.add_argument('--interval', type=float, default=2.0, help='seconds between captures')
    parser.add_argument('--source', default=svc.DEFAULT_SOURCE_NAME, help='video_source -n name')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Saving pairs to {args.output_dir}/ as left-NNNN.png / right-NNNN.png every {args.interval}s')

    count = 0
    last_save = 0.0
    with svc.StereoVideoCapture(args.source) as capture:
        print('Waiting for synced image pairs...')
        try:
            while True:
                time.sleep(0.01)
                if time.time() - last_save < args.interval:
                    continue
                frame = capture.latest()
                if frame is None:
                    continue
                last_save = time.time()
                lfname = f'{args.output_dir}/left-{count:04d}.png'
                rfname = f'{args.output_dir}/right-{count:04d}.png'
                cv2.imwrite(lfname, frame.left)
                cv2.imwrite(rfname, frame.right)
                print(f'Saved pair {count}: {lfname}, {rfname}')
                count += 1
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
