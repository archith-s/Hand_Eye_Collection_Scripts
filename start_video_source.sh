#!/usr/bin/env bash
# Starts the DecklinkVideoGstreamerClient stereo video source that every
# image-recording script here (and stereo_video_ros_bridge.py) reads from.
# Replaces `ros2 launch dvrk_video decklink_stereo_goovis.launch.py` -- the
# two can NOT run at the same time (the DeckLink cards are exclusive).
#
# Device mapping: on this rig DeckLink device 0 is the physical LEFT eye and
# device 1 the physical RIGHT eye, so LEFT=0 RIGHT=1. dvrk_video's launch
# files default to the opposite (left_device 1, right_device 0), which
# swapped the eyes: the stereo calibration came out with the right camera
# ~5.4 mm to the LEFT of the left one (positive Tx / T[0]). Datasets recorded
# before this fix (MarkerlessCalibDatasets run1-run4) have left/right swapped
# and were calibrated as such -- don't mix them with new recordings.
# Override with LEFT=.. RIGHT=.. if needed.

set -e

CLIENT_DIR=${CLIENT_DIR:-$HOME/DecklinkVideoGstreamerClient}
LEFT=${LEFT:-0}
RIGHT=${RIGHT:-1}
MODE=${MODE:-1080i5994}

if [[ ! -x "$CLIENT_DIR/build/video_source" ]]; then
    echo "Building $CLIENT_DIR (with python bindings)..."
    cmake -S "$CLIENT_DIR" -B "$CLIENT_DIR/build" -DBUILD_PYTHON=ON -DPython_EXECUTABLE=/usr/bin/python3
    cmake --build "$CLIENT_DIR/build" -j
fi

exec "$CLIENT_DIR/build/video_source" -l "$LEFT" -r "$RIGHT" --connection sdi --mode "$MODE" "$@"
