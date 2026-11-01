#!/usr/bin/env python3

# Quick go/no-go check for hand_eye_gt_raw_corrections.json files: reports
# whether a dataset's manual corrections are complete and self-consistent
# enough to trust before running 3d_pose_alignment.py's final hand-eye
# update, and calls out specific frames worth redoing rather than an
# all-or-nothing verdict. Meant to be run right after finishing a dataset,
# or on everything right before you push an update.
#
# Thresholds below are calibrated against real sessions from this project:
# a genuinely careful pass came out around 3-4mm / 2-3deg std, while a
# rushed one came out around 20mm+ / 10deg+ with individual frames 30-80mm
# / 30-37deg off. They're heuristics, not hard science -- treat borderline
# results as "look at the outlier frames yourself", not gospel.

import argparse
import json
import os
import sys

import numpy
from scipy.spatial.transform import Rotation

GT_JSON_NAME = 'hand_eye_gt_raw.json'
CORRECTIONS_JSON_NAME = 'hand_eye_gt_raw_corrections.json'

GOOD_STD_TRANS_MM = 8.0
GOOD_STD_ROT_DEG = 5.0
OUTLIER_TRANS_MM = 30.0
OUTLIER_ROT_DEG = 15.0


def average_transforms(matrices):
    # Same as 3d_pose_alignment.py's average_transforms: translation by
    # arithmetic mean, rotation by SVD-orthonormalized mean rotation matrix.
    translations = numpy.array([m[0:3, 3] for m in matrices])
    rotations = numpy.array([m[0:3, 0:3] for m in matrices])

    t_avg = translations.mean(axis=0)
    U, _, Vt = numpy.linalg.svd(rotations.sum(axis=0))
    R_avg = U @ Vt
    if numpy.linalg.det(R_avg) < 0:
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


def check_dataset(path: str):
    name = os.path.basename(os.path.normpath(path))
    gt_path = os.path.join(path, GT_JSON_NAME)
    corr_path = os.path.join(path, CORRECTIONS_JSON_NAME)

    if not os.path.exists(gt_path):
        print(f'{name}: not a hand_eye_gt_raw.py output directory (no {GT_JSON_NAME})')
        return None
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    total_frames = len(gt['records'])

    if not os.path.exists(corr_path):
        print(f'{name}: NOT STARTED (0/{total_frames} frames reviewed)')
        return None
    with open(corr_path, 'r') as f:
        saved = json.load(f)
    corrections = saved.get('corrections', {})
    skipped = set(saved.get('skipped', []))
    reviewed = len(corrections) + len(skipped)

    if reviewed < total_frames:
        print(f'{name}: IN PROGRESS ({reviewed}/{total_frames} frames reviewed -- '
              f'{len(corrections)} corrected, {len(skipped)} skipped)')
        return None
    if not corrections:
        print(f'{name}: all {total_frames} frames skipped -- nothing to combine, redo at least a few')
        return False

    frame_indices = sorted(int(k) for k in corrections.keys())
    matrices = [numpy.array(corrections[str(i)]['cam_T_base_estimate']).reshape((4, 4)) for i in frame_indices]
    avg, diag = average_transforms(matrices)

    outliers = []
    for i, M in zip(frame_indices, matrices):
        diff = numpy.linalg.inv(avg) @ M
        trans_mm = numpy.linalg.norm(diff[0:3, 3]) * 1000.0
        rot_deg = Rotation.from_matrix(diff[0:3, 0:3]).magnitude() * 180.0 / numpy.pi
        if trans_mm > OUTLIER_TRANS_MM or rot_deg > OUTLIER_ROT_DEG:
            outliers.append((i, trans_mm, rot_deg))

    good = (diag['trans_std_mm'] <= GOOD_STD_TRANS_MM
            and diag['rot_std_deg'] <= GOOD_STD_ROT_DEG
            and not outliers)
    verdict = 'READY' if good else 'NEEDS REVIEW'
    print(f'{name}: {verdict}  ({len(corrections)} corrected, {len(skipped)} skipped)  '
          f'spread std {diag["trans_std_mm"]:.1f}mm/{diag["rot_std_deg"]:.1f}deg  '
          f'(max {diag["trans_max_mm"]:.1f}mm/{diag["rot_max_deg"]:.1f}deg)')
    if outliers:
        outliers.sort(key=lambda r: -r[2])
        print(f'  outlier frames (over {OUTLIER_TRANS_MM:.0f}mm or {OUTLIER_ROT_DEG:.0f}deg '
              f'from this dataset\'s own average):')
        for i, tr, ro in outliers:
            print(f'    frame {i}: {tr:.1f}mm / {ro:.1f}deg')
    return good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'datasets', type=str, nargs='+',
        help='one or more hand_eye_gt_raw.py output directories to check',
    )
    args = parser.parse_args(sys.argv[1:])

    results = [check_dataset(path) for path in args.datasets]

    print()
    if any(r is None for r in results):
        print('Not all datasets are finished yet -- nothing more to say until they are.')
    elif all(r is True for r in results):
        print('All datasets look self-consistent -- OK to run the joint update.')
    else:
        print('At least one dataset needs review before pushing -- see outlier frames above.')


if __name__ == '__main__':
    main()
