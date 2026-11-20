#!/usr/bin/env python3
"""Compile the per-run markerless calibration evaluations into one cumulative PDF.

Reads, for every run directory under ROOT (default ~/MarkerlessCalibDatasets/run*):
  eval_gt_hand_eye_dataset_markerless_dr_comparison.json   hand-eye vs diff-render vs GT
  gt_segmentation_dataset/segmentation_eval.json           GT clevis vs hand segmentation
      (or gt_segmentation_dataset/recovered_from_pdf/segmentation_eval.json)
and frame-15 overlay panels from each run's eval PDF (via pdfimages, if available).

Usage: python3 cumulative_markerless_eval.py [ROOT] [-o OUT.pdf]
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

HE = "hand-eye (marker only)"
DR = "diff-render (markerless)"
C_HE, C_DR, C_PNP = "#2a6fb0", "#e07b20", "#9a9a9a"
A4L = (11.69, 8.27)


def load_runs(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "run*"))):
        ev = os.path.join(d, "eval_gt_hand_eye_dataset_markerless_dr_comparison.json")
        if not os.path.isfile(ev):
            continue
        r = {"name": os.path.basename(d), "dir": d, "eval": json.load(open(ev))}
        seg, src = None, None
        for p, s in [("gt_segmentation_dataset/segmentation_eval.json", "original"),
                     ("gt_segmentation_dataset/recovered_from_pdf/segmentation_eval.json",
                      "recovered from PDF (2-3 dp)")]:
            if os.path.isfile(os.path.join(d, p)):
                seg, src = json.load(open(os.path.join(d, p))), s
                break
        r["seg"], r["seg_src"] = seg, src
        runs.append(r)
    return runs


def t_err_mm(A, B):
    return 1000 * np.linalg.norm(np.asarray(A)[:3, 3] - np.asarray(B)[:3, 3])


def stats(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, np.median(x), x.min(), x.max()


def fmt_ms(x, p=2):
    m, s, *_ = stats(x)
    return f"{m:.{p}f} ± {s:.{p}f}"


def table(ax, rows, cols, col_w=None, fs=8.5, header_color="#dde6f0", scale=1.4, bold_last=0):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center",
                 colWidths=col_w)
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, scale)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#bbbbbb")
        if i == 0:
            c.set_facecolor(header_color)
            c.set_text_props(weight="bold")
        elif bold_last and i > len(rows) - bold_last:
            c.set_facecolor("#f3f3f3")
            c.set_text_props(weight="bold")
    return t


def page_title(fig, title, sub=None):
    fig.text(0.04, 0.955, title, fontsize=15, weight="bold")
    if sub:
        fig.text(0.04, 0.925, sub, fontsize=9, color="#555555")


def note(fig, text, y=0.03):
    import textwrap
    fig.text(0.04, y, textwrap.fill(text, 190), fontsize=7.5, color="#444444", va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.expanduser("~/MarkerlessCalibDatasets"))
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    runs = load_runs(a.root)
    if not runs:
        raise SystemExit(f"no runs found under {a.root}")
    out = a.out or os.path.join(a.root, f"cumulative_eval_{len(runs)}runs.pdf")
    names = [r["name"] for r in runs]
    N = len(runs)

    # ---------- derived per-run numbers ----------
    for r in runs:
        e = r["eval"]
        gt = e["cam_T_base_ground_truth"]
        r["pnp_base_mm"] = t_err_mm(e["cam_T_base_pnp_init"], gt)
        r["he_vs_dr_base_mm"] = t_err_mm(e["cam_T_base_marker_only"], e["cam_T_base_diff_render"])
        tf = e["test_frame_results"]
        for m, k in [(HE, "he"), (DR, "dr")]:
            r[f"{k}_t"] = [f[m]["trans_err_mm"] for f in tf]
            r[f"{k}_r"] = [f[m]["rot_err_deg"] for f in tf]
            r[f"{k}_iou"] = [f[m]["iou_vs_gt"] for f in tf]
            r[f"{k}_add"] = [f[m]["add_mm"] for f in tf]
        r["test_ids"] = [f["frame_id"] for f in tf]

    def ov(r, m, k):
        return r["eval"]["overall_on_test"][m][k]

    pool = {k: np.concatenate([r[k] for r in runs]) for k in
            ["he_t", "he_r", "he_iou", "he_add", "dr_t", "dr_r", "dr_iou", "dr_add"]}

    with PdfPages(out) as pdf:
        # ================= PAGE 1: overview =================
        fig = plt.figure(figsize=A4L)
        page_title(fig, f"Markerless hand-eye calibration — cumulative evaluation over {N} runs",
                   f"Source: {a.root}  ({', '.join(names)}).  Ground truth = marker registration + 3D pose "
                   "alignment; used only for scoring.")
        rows = []
        for r in runs:
            e = r["eval"]
            rows.append([r["name"], e["generated"].replace("T", " "),
                         f"{len(e['train_frames'])} / {len(e['test_frames'])}",
                         f"{len(e['tip_found'])}", f"{len(e['inliers'])}",
                         f"{e['mask_passes']}", f"{e['fit_runtime_s']:.0f}",
                         e["runtime"].get("device", "")])
        ax = fig.add_axes([0.04, 0.68, 0.92, 0.22])
        table(ax, rows, ["run", "eval generated", "train / held-out", "tip found",
                         "inliers used", "DR passes", "fit time (s)", "device"],
              col_w=[0.07, 0.17, 0.12, 0.09, 0.1, 0.08, 0.1, 0.2])

        fig.text(0.04, 0.635, "Whole calibration vs ground truth on the held-out frames — mean ± std across runs "
                 f"(n = {N} runs)", fontsize=11, weight="bold")
        rows = []
        for lbl, get in [("PnP init (no prior)",
                          lambda r: (r["eval"]["pnp_init_vs_gt_on_test"]["mean_clevis_displacement_mm"],
                                     r["eval"]["pnp_init_vs_gt_on_test"]["rotation_err_deg"],
                                     r["pnp_base_mm"], np.nan)),
                         (HE, lambda r: (ov(r, HE, "mean_clevis_displacement_mm"), ov(r, HE, "rotation_err_deg"),
                                         ov(r, HE, "base_origin_err_mm"), ov(r, HE, "eval_mean_iou_vs_gt"))),
                         (DR, lambda r: (ov(r, DR, "mean_clevis_displacement_mm"), ov(r, DR, "rotation_err_deg"),
                                         ov(r, DR, "base_origin_err_mm"), ov(r, DR, "eval_mean_iou_vs_gt")))]:
            v = np.array([get(r) for r in runs])
            rows.append([lbl, fmt_ms(v[:, 0]), f"{np.median(v[:, 0]):.2f}", fmt_ms(v[:, 1]),
                         fmt_ms(v[:, 2]), "—" if np.isnan(v[0, 3]) else fmt_ms(v[:, 3], 3)])
        ax = fig.add_axes([0.04, 0.47, 0.92, 0.15])
        table(ax, rows, ["calibration", "clevis displ. (mm)", "median (mm)", "rotation err (deg)",
                         "base-origin err (mm)", "silhouette IoU vs GT"],
              col_w=[0.22, 0.15, 0.1, 0.15, 0.16, 0.16])

        # head-to-head
        dr_better_t = sum(ov(r, DR, "mean_clevis_displacement_mm") < ov(r, HE, "mean_clevis_displacement_mm")
                          for r in runs)
        dr_better_iou = sum(ov(r, DR, "eval_mean_iou_vs_gt") > ov(r, HE, "eval_mean_iou_vs_gt") for r in runs)
        dr_better_r = sum(ov(r, DR, "rotation_err_deg") < ov(r, HE, "rotation_err_deg") for r in runs)
        pnp_gain = [r["eval"]["pnp_init_vs_gt_on_test"]["mean_clevis_displacement_mm"]
                    / ov(r, DR, "mean_clevis_displacement_mm") for r in runs]
        fig.text(0.04, 0.43, "Key findings", fontsize=11, weight="bold")
        lines = [
            f"• Pooled over all {len(pool['he_t'])} held-out frames: clevis translation error  hand-eye "
            f"{pool['he_t'].mean():.2f} mm (median {np.median(pool['he_t']):.2f})  vs  diff-render "
            f"{pool['dr_t'].mean():.2f} mm (median {np.median(pool['dr_t']):.2f}).",
            f"• Rotation error: hand-eye {fmt_ms([ov(r, HE, 'rotation_err_deg') for r in runs])} deg  vs  "
            f"diff-render {fmt_ms([ov(r, DR, 'rotation_err_deg') for r in runs])} deg — the marker calibration is "
            f"better in rotation in {N - dr_better_r}/{N} runs.",
            f"• Translation: diff-render beats the marker calibration in {dr_better_t}/{N} runs "
            f"(run-level), and its rendered silhouette overlaps GT better (higher IoU) in {dr_better_iou}/{N} runs "
            f"(pooled IoU {pool['dr_iou'].mean():.3f} vs {pool['he_iou'].mean():.3f}).",
            f"• Diff-render refinement cuts the PnP-init clevis error by {np.mean(pnp_gain):.1f}x on average "
            f"(range {min(pnp_gain):.1f}x–{max(pnp_gain):.1f}x).",
            f"• Run-to-run spread is large for both methods: hand-eye clevis error ranges "
            f"{min(ov(r, HE, 'mean_clevis_displacement_mm') for r in runs):.1f}–"
            f"{max(ov(r, HE, 'mean_clevis_displacement_mm') for r in runs):.1f} mm, diff-render "
            f"{min(ov(r, DR, 'mean_clevis_displacement_mm') for r in runs):.1f}–"
            f"{max(ov(r, DR, 'mean_clevis_displacement_mm') for r in runs):.1f} mm.",
        ]
        # anomalies
        for r in runs:
            e = r["eval"]
            if len(e["inliers"]) < len(e["train_frames"]):
                miss = sorted(set(e["train_frames"]) - set(e["inliers"]))
                lines.append(f"• {r['name']}: only {len(e['inliers'])}/{len(e['train_frames'])} training frames "
                             f"used (no tool tip / rejected: {miss}); calibration frame 14 status: "
                             f"\"{e['calibration_frame']['status']}\".")
        rec = [r["name"] for r in runs if r["seg_src"] and "recovered" in r["seg_src"]]
        if rec:
            lines.append(f"• Segmentation-vs-GT numbers for {', '.join(rec)} were recovered from the PDF "
                         "(rounded to 2–3 decimals); originals were deleted.")
        fig.text(0.05, 0.415, "\n".join(lines), fontsize=8.6, va="top", linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)

        # ================= PAGE 2: per-run whole-calibration table + bars =================
        fig = plt.figure(figsize=A4L)
        page_title(fig, "Per-run whole-calibration error on held-out frames",
                   "Each calibration applied to the held-out frames' kinematics; nothing is fit per frame.")
        rows = []
        for r in runs:
            e = r["eval"]
            rows.append([r["name"],
                         f"{e['pnp_init_vs_gt_on_test']['mean_clevis_displacement_mm']:.2f}",
                         f"{e['pnp_init_vs_gt_on_test']['rotation_err_deg']:.2f}",
                         f"{ov(r, HE, 'mean_clevis_displacement_mm'):.2f}", f"{ov(r, HE, 'rotation_err_deg'):.2f}",
                         f"{ov(r, HE, 'base_origin_err_mm'):.2f}", f"{ov(r, HE, 'eval_mean_iou_vs_gt'):.3f}",
                         f"{ov(r, DR, 'mean_clevis_displacement_mm'):.2f}", f"{ov(r, DR, 'rotation_err_deg'):.2f}",
                         f"{ov(r, DR, 'base_origin_err_mm'):.2f}", f"{ov(r, DR, 'eval_mean_iou_vs_gt'):.3f}",
                         f"{r['he_vs_dr_base_mm']:.2f}"])
        cols_k = [("mean_clevis_displacement_mm", 2), ("rotation_err_deg", 2), ("base_origin_err_mm", 2),
                  ("eval_mean_iou_vs_gt", 3)]
        pn = [r["eval"]["pnp_init_vs_gt_on_test"] for r in runs]
        mean_row = ["mean",
                    f"{np.mean([p['mean_clevis_displacement_mm'] for p in pn]):.2f}",
                    f"{np.mean([p['rotation_err_deg'] for p in pn]):.2f}"]
        for m in (HE, DR):
            mean_row += [f"{np.mean([ov(r, m, k) for r in runs]):.{p}f}" for k, p in cols_k]
        mean_row.append(f"{np.mean([r['he_vs_dr_base_mm'] for r in runs]):.2f}")
        rows.append(mean_row)
        ax = fig.add_axes([0.03, 0.66, 0.94, 0.24])
        table(ax, rows, ["run", "PnP\nclevis mm", "PnP\nrot °", "HE\nclevis mm", "HE\nrot °", "HE\nbase mm",
                         "HE\nIoU", "DR\nclevis mm", "DR\nrot °", "DR\nbase mm", "DR\nIoU", "HE↔DR\nbase mm"],
              fs=8, scale=1.9, bold_last=1)

        x = np.arange(N)
        w = 0.38
        metrics = [("mean_clevis_displacement_mm", "clevis displacement (mm)"),
                   ("rotation_err_deg", "rotation error (deg)"),
                   ("base_origin_err_mm", "base-origin error (mm)"),
                   ("eval_mean_iou_vs_gt", "silhouette IoU vs GT (↑)")]
        for i, (k, lab) in enumerate(metrics):
            ax = fig.add_axes([0.06 + i * 0.235, 0.1, 0.19, 0.46])
            he = [ov(r, HE, k) for r in runs]
            dr = [ov(r, DR, k) for r in runs]
            ax.bar(x - w / 2, he, w, color=C_HE, label="hand-eye (marker)")
            ax.bar(x + w / 2, dr, w, color=C_DR, label="diff-render (markerless)")
            ax.axhline(np.mean(he), color=C_HE, ls="--", lw=1)
            ax.axhline(np.mean(dr), color=C_DR, ls="--", lw=1)
            ax.set_xticks(x, names)
            ax.set_title(lab, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.legend(fontsize=7.5, loc="upper left")
        note(fig, "Dashed lines = mean across runs.  HE = hand-eye (marker only), DR = diff-render (markerless), "
                  "PnP = DR's PnP initialisation.  HE↔DR base mm = distance between the two calibrations' base "
                  "origins (independent of GT).")
        pdf.savefig(fig)
        plt.close(fig)

        # ================= PAGE 3: pooled held-out frame distributions =================
        fig = plt.figure(figsize=A4L)
        page_title(fig, f"Held-out frames pooled across runs ({len(pool['he_t'])} frames)",
                   "Per-frame errors of each calibration applied to that frame's kinematics.")
        panels = [("t", "clevis translation error (mm)"), ("add", "ADD (mm)"), ("iou", "silhouette IoU vs GT (↑)")]
        rng = np.random.default_rng(0)
        for i, (k, lab) in enumerate(panels):
            ax = fig.add_axes([0.06 + i * 0.32, 0.47, 0.26, 0.39])
            data = [pool[f"he_{k}"], pool[f"dr_{k}"]]
            bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
            for patch, c in zip(bp["boxes"], (C_HE, C_DR)):
                patch.set_facecolor(c)
                patch.set_alpha(0.35)
            markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
            for j, m in enumerate(("he", "dr")):
                for ri, r in enumerate(runs):
                    v = r[f"{m}_{k}"]
                    ax.scatter(j + 1 + rng.uniform(-0.15, 0.15, len(v)), v, s=16,
                               marker=markers[ri % len(markers)], color=(C_HE, C_DR)[j],
                               edgecolor="k", lw=0.4, label=r["name"] if j == 0 else None)
            ax.set_xticks([1, 2], ["hand-eye\n(marker)", "diff-render\n(markerless)"], fontsize=8.5)
            ax.set_title(lab, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7, title="run (marker)", title_fontsize=7)
        rows = []
        for k, lab, p in [("t", "clevis trans. err (mm)", 2), ("r", "rotation err (deg)", 2),
                          ("add", "ADD (mm)", 2), ("iou", "IoU vs GT", 3)]:
            for m, ml in (("he", "hand-eye"), ("dr", "diff-render")):
                mu, sd, md, lo, hi = stats(pool[f"{m}_{k}"])
                rows.append([lab, ml, f"{mu:.{p}f}", f"{sd:.{p}f}", f"{md:.{p}f}", f"{lo:.{p}f}", f"{hi:.{p}f}"])
        ax = fig.add_axes([0.1, 0.04, 0.8, 0.33])
        table(ax, rows, ["metric", "calibration", "mean", "std", "median", "min", "max"], fs=8.5, scale=1.35)
        pdf.savefig(fig)
        plt.close(fig)

        # ================= PAGE 4: per-frame held-out table =================
        fig = plt.figure(figsize=A4L)
        page_title(fig, "Held-out frame results, all runs",
                   "Rotation error equals the calibration's rotation error and is identical for every frame "
                   "of a run.")
        rows = []
        for r in runs:
            for i, fid in enumerate(r["test_ids"]):
                rows.append([r["name"], str(fid),
                             f"{r['he_t'][i]:.2f}", f"{r['he_r'][i]:.2f}", f"{r['he_add'][i]:.2f}",
                             f"{r['he_iou'][i]:.3f}",
                             f"{r['dr_t'][i]:.2f}", f"{r['dr_r'][i]:.2f}", f"{r['dr_add'][i]:.2f}",
                             f"{r['dr_iou'][i]:.3f}"])
        ax = fig.add_axes([0.06, 0.04, 0.88, 0.86])
        table(ax, rows, ["run", "frame", "HE trans mm", "HE rot °", "HE ADD mm", "HE IoU",
                         "DR trans mm", "DR rot °", "DR ADD mm", "DR IoU"], fs=7.8, scale=1.12)
        pdf.savefig(fig)
        plt.close(fig)

        # ================= PAGE 5: calibration frame (SurgRIPE) + error breakdown =================
        fig = plt.figure(figsize=A4L)
        page_title(fig, "Last training frame (frame 14) as a 6D pose estimate — SurgRIPE metrics",
                   f"Model diameter {runs[0]['eval']['calibration_frame']['model_diameter_mm']:.2f} mm; "
                   "pass thresholds: ADD/ADD-S < 10% diam., 2D proj < 5 px, 5 mm / 5°.")
        keys = [("trans_error", "trans. err (mm)", 2), ("rot_error", "rot. err (deg)", 2),
                ("ADD-distance", "ADD (mm)", 2), ("ADDS-distance", "ADD-S (mm)", 2),
                ("proj2d_err_px", "2D proj err (px)", 1), ("add_auc_0_diam", "ADD AUC", 3),
                ("avg_acc_0_10mm", "Avg Acc 0-10mm", 3), ("iou_vs_gt", "IoU vs GT", 3)]
        rows = []
        for r in runs:
            cf = r["eval"]["calibration_frame"]
            for m, ml in ((HE, "HE"), (DR, "DR")):
                s = cf["surgripe_metrics"][m]
                passes = "".join("✓" if s[p] else "·" for p in ("add", "add-s", "proj2d", "cmd5"))
                rows.append([r["name"] + ("*" if "not used" in cf["status"] else ""), ml]
                            + [f"{s[k]:.{p}f}" for k, _, p in keys] + [passes])
        for m, ml in ((HE, "HE"), (DR, "DR")):
            rows.append(["mean", ml] + [f"{np.mean([r['eval']['calibration_frame']['surgripe_metrics'][m][k] for r in runs]):.{p}f}"
                                        for k, _, p in keys] + [""])
        ax = fig.add_axes([0.03, 0.5, 0.94, 0.4])
        table(ax, rows, ["run", "calib"] + [l for _, l, _ in keys] + ["pass †"], fs=7.8,
              scale=1.45, bold_last=2)

        fig.text(0.04, 0.44, "Diff-render error breakdown on frame 14 (camera frame)", fontsize=11, weight="bold")
        bk = [("depth_gt_mm", "depth GT (mm)", 1), ("depth_est_mm", "depth DR (mm)", 1),
              ("trans_along_sight_mm", "along sight (mm)", 2), ("trans_sideways_mm", "sideways (mm)", 2),
              ("rot_about_clevis_axis_deg", "roll about clevis (°)", 2), ("rot_about_sight_deg", "about sight (°)", 2),
              ("projected_size_gt_px", "proj size GT (px)", 0), ("projected_size_est_px", "proj size DR (px)", 0)]
        rows = []
        for r in runs:
            b = r["eval"]["calibration_frame"]["error_breakdown_diff_render"]
            rows.append([r["name"]] + [f"{b[k]:.{p}f}" for k, _, p in bk])
        rows.append(["mean |·|"] + [f"{np.mean([abs(r['eval']['calibration_frame']['error_breakdown_diff_render'][k]) for r in runs]):.{p}f}"
                                    for k, _, p in bk])
        ax = fig.add_axes([0.06, 0.16, 0.88, 0.26])
        table(ax, rows, ["run"] + [l for _, l, _ in bk], fs=8, scale=1.5, bold_last=1)
        along = [abs(r["eval"]["calibration_frame"]["error_breakdown_diff_render"]["trans_along_sight_mm"]) for r in runs]
        side = [abs(r["eval"]["calibration_frame"]["error_breakdown_diff_render"]["trans_sideways_mm"]) for r in runs]
        note(fig, f"Along-sight (depth) error dominates: mean |along| {np.mean(along):.1f} mm vs sideways "
                  f"{np.mean(side):.1f} mm. Negative along-sight = DR puts the clevis closer to the camera than GT. "
                  "† pass flags in order ADD / ADD-S / 2D proj / 5mm-5°: ✓ pass, · fail (single image -> 0/1).  * frame 14 had no tool tip found in that run and "
                  "was not used in the fit (scored as held-out).", y=0.05)
        pdf.savefig(fig)
        plt.close(fig)

        # ================= PAGE 6: generated masks vs hand segmentation =================
        fig = plt.figure(figsize=A4L)
        page_title(fig, "Diff-render generated masks vs hand segmentation (training frames)",
                   "IoU of DR final-pass mask vs drawn mask; pose errors = pose fit to generated mask vs pose fit "
                   "to drawn mask (pitch pin, camera frame).")
        mk = [("iou", "IoU", 3), ("iou_dr_region", "IoU (DR rgn)", 3), ("rot_err_deg", "rot err (°)", 2),
              ("trans_err_mm", "trans err (mm)", 2), ("trans_err_inplane_mm", "in-plane (mm)", 2),
              ("trans_err_depth_mm", "depth (mm)", 2)]
        rows, allmb = [], {k: [] for k, _, _ in mk}
        for r in runs:
            fr = r["eval"]["mask_baseline"]["frames"]
            for k, _, _ in mk:
                allmb[k] += [f[k] for f in fr]
            rows.append([r["name"], str(len(fr))] + [fmt_ms([f[k] for f in fr], p) for k, _, p in mk])
        rows.append(["pooled", str(len(allmb["iou"]))] + [fmt_ms(allmb[k], p) for k, _, p in mk])
        rows.append(["pooled median", ""] + [f"{np.median(allmb[k]):.{p}f}" for k, _, p in mk])
        ax = fig.add_axes([0.04, 0.6, 0.92, 0.3])
        table(ax, rows, ["run", "frames"] + [l for _, l, _ in mk], fs=8.3, scale=1.6, bold_last=2)

        # also hand-fit vs GT from mask_baseline
        for i, (k, lab) in enumerate([("iou", "IoU (whole image)"), ("iou_dr_region", "IoU (DR loss region)"),
                                      ("trans_err_depth_mm", "depth error (mm)"),
                                      ("trans_err_inplane_mm", "in-plane error (mm)")]):
            ax = fig.add_axes([0.06 + i * 0.235, 0.1, 0.19, 0.4])
            data = [[f[k] for f in r["eval"]["mask_baseline"]["frames"]] for r in runs]
            ax.boxplot(data, widths=0.55, showfliers=True)
            ax.set_xticks(range(1, N + 1), names, fontsize=8)
            ax.set_title(lab, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        note(fig, "'DR rgn' = only pixels the diff-render loss uses (jaw / shaft / ring-prong zones excluded). "
                  "Values are mean ± std over the evaluated frames (10 per run, evenly spaced among training frames). "
                  "Depth is only weakly observable from a silhouette.")
        pdf.savefig(fig)
        plt.close(fig)

        # ================= PAGE 7: GT vs hand segmentation =================
        seg_runs = [r for r in runs if r["seg"]]
        if seg_runs:
            fig = plt.figure(figsize=A4L)
            page_title(fig, "Ground-truth clevis vs hand segmentation (gt_segmentation_dataset)",
                       "IoU = GT clevis silhouette vs drawn mask; errors = GT clevis pose vs pose best fitting "
                       "the drawn mask.")
            sk = [("iou", "IoU", 3), ("rot_err_deg", "rot err (°)", 2), ("trans_err_mm", "trans err (mm)", 2),
                  ("trans_err_inplane_mm", "in-plane (mm)", 2), ("trans_err_depth_mm", "depth (mm)", 2)]
            rows, alls = [], {k: [] for k, _, _ in sk}
            for r in seg_runs:
                res = r["seg"]["results"]
                for k, _, _ in sk:
                    alls[k] += [f[k] for f in res]
                rows.append([r["name"], str(len(res)), r["seg_src"]] + [fmt_ms([f[k] for f in res], p) for k, _, p in sk])
            rows.append(["pooled", str(len(alls["iou"])), ""] + [fmt_ms(alls[k], p) for k, _, p in sk])
            rows.append(["pooled median", "", ""] + [f"{np.median(alls[k]):.{p}f}" for k, _, p in sk])
            ax = fig.add_axes([0.03, 0.6, 0.94, 0.3])
            table(ax, rows, ["run", "frames", "source"] + [l for _, l, _ in sk], fs=8.2, scale=1.6, bold_last=2,
                  col_w=[0.08, 0.06, 0.17] + [0.13] * 5)
            for i, (k, lab) in enumerate([("iou", "IoU"), ("rot_err_deg", "rotation error (deg)"),
                                          ("trans_err_inplane_mm", "in-plane error (mm)"),
                                          ("trans_err_depth_mm", "depth error (mm)")]):
                ax = fig.add_axes([0.06 + i * 0.235, 0.1, 0.19, 0.4])
                ax.boxplot([[f[k] for f in r["seg"]["results"]] for r in seg_runs], widths=0.55)
                ax.set_xticks(range(1, len(seg_runs) + 1), [r["name"] for r in seg_runs], fontsize=8)
                ax.set_title(lab, fontsize=9)
                ax.grid(axis="y", alpha=0.3)
            note(fig, "This measures how well a human-drawn clevis mask agrees with the GT pose: a ceiling on what "
                      "silhouette-based fitting can recover. Large rotation errors reflect the near-symmetric clevis "
                      "silhouette (roll is barely observable).")
            pdf.savefig(fig)
            plt.close(fig)

        # ================= PAGE 8: runtime =================
        fig = plt.figure(figsize=A4L)
        page_title(fig, "Calibration runtime", f"Device: {runs[0]['eval']['runtime'].get('device', '?')}")
        rows, comp = [], []
        for r in runs:
            rt = r["eval"]["runtime"]
            pi = rt["per_image"].values()
            load = np.mean([p["load_s"] + p["stereo_box_s"] for p in pi])
            mask = np.mean([p["mask_s"] for p in rt["per_image"].values()])
            j = rt["joint"]
            dr = sum(j["dr_passes_s"])
            comp.append([sum(p["load_s"] + p["stereo_box_s"] for p in rt["per_image"].values()),
                         sum(p["mask_s"] for p in rt["per_image"].values()), j["pnp_init_s"], dr,
                         j["outlier_check_s"] + j.get("refit_s", 0), rt.get("other_s", 0)])
            rows.append([r["name"], str(rt["n_images"]), str(rt["n_images_fit"]), f"{load:.2f}", f"{mask:.2f}",
                         f"{j['pnp_init_s']:.2f}", f"{dr:.1f}", f"{rt['joint_total_s']:.1f}",
                         f"{rt['per_image_total_s']:.1f}", f"{rt['total_s']:.1f}", f"{rt.get('startup_s', 0):.1f}"])
        tot = [r["eval"]["runtime"]["total_s"] for r in runs]
        rows.append(["mean", "", "", "", "", "", "", "", "", f"{np.mean(tot):.1f}", ""])
        ax = fig.add_axes([0.03, 0.66, 0.94, 0.24])
        table(ax, rows, ["run", "images", "fit imgs", "load+box /img", "mask /img", "PnP init", "DR passes",
                         "joint total", "per-image total", "TOTAL", "startup"], fs=8, scale=1.6, bold_last=1)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.48])
        comp = np.array(comp)
        labs = ["load + stereo box", "masks (GrabCut)", "PnP init", "DR passes", "outlier check + refit",
                "python overhead"]
        cols = ["#8fb3d9", "#e3a15c", "#bbbbbb", "#c0504d", "#7f9f5f", "#dddddd"]
        left = np.zeros(N)
        for i in range(comp.shape[1]):
            ax.barh(names, comp[:, i], left=left, color=cols[i], label=labs[i])
            left += comp[:, i]
        ax.invert_yaxis()
        ax.set_xlabel("seconds")
        ax.legend(fontsize=8, ncol=3, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        ax.set_title("Where the fit time goes", fontsize=10)
        note(fig, "Mask generation dominates the runtime; the joint differentiable-rendering optimisation itself "
                  f"takes ~{np.mean([r['eval']['runtime']['joint_total_s'] for r in runs]):.0f} s.")
        pdf.savefig(fig)
        plt.close(fig)

        # ================= PAGE 9: calibration matrices =================
        fig = plt.figure(figsize=A4L)
        page_title(fig, "cam_T_base per run (translation in mm, rotation as axis-angle deg)",
                   "Differences from GT for each calibration.")
        rows = []

        def rotdeg(A, B):
            R = np.asarray(A)[:3, :3].T @ np.asarray(B)[:3, :3]
            return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        for r in runs:
            e = r["eval"]
            gt = np.asarray(e["cam_T_base_ground_truth"])
            for key, lab in [("cam_T_base_ground_truth", "GT"), ("cam_T_base_marker_only", "hand-eye"),
                             ("cam_T_base_diff_render", "diff-render"), ("cam_T_base_pnp_init", "PnP init")]:
                T = np.asarray(e[key])
                t = T[:3, 3] * 1000
                d = (T[:3, 3] - gt[:3, 3]) * 1000
                rows.append([r["name"], lab, f"{t[0]:.1f}", f"{t[1]:.1f}", f"{t[2]:.1f}",
                             "—" if lab == "GT" else f"{d[0]:+.1f}", "—" if lab == "GT" else f"{d[1]:+.1f}",
                             "—" if lab == "GT" else f"{d[2]:+.1f}",
                             "—" if lab == "GT" else f"{np.linalg.norm(d):.1f}",
                             "—" if lab == "GT" else f"{rotdeg(gt, T):.2f}"])
        ax = fig.add_axes([0.06, 0.04, 0.88, 0.86])
        table(ax, rows, ["run", "calibration", "tx", "ty", "tz", "Δtx", "Δty", "Δtz", "|Δt| mm", "Δrot °"],
              fs=8, scale=1.2)
        pdf.savefig(fig)
        plt.close(fig)

        # ================= PAGE 10: frame-15 overlays =================
        if shutil.which("pdfimages"):
            tmp = tempfile.mkdtemp()
            imgs = []
            for r in runs:
                src = os.path.join(r["dir"], "eval_gt_hand_eye_dataset_markerless_dr_comparison.pdf")
                pre = os.path.join(tmp, r["name"])
                subprocess.run(["pdfimages", "-f", "5", "-l", "5", "-png", src, pre], check=False,
                               capture_output=True)
                pics = sorted(glob.glob(pre + "-*.png"))
                imgs.append(pics[:2] if len(pics) >= 2 else None)
            if any(imgs):
                fig = plt.figure(figsize=A4L)
                page_title(fig, "Held-out frame 15 overlays per run",
                           "Left: hand-eye (marker only) vs GT.  Right: diff-render (markerless) vs GT.  "
                           "(Panels as embedded in each run's eval PDF.)")
                h = 0.84 / N
                for i, (r, pics) in enumerate(zip(runs, imgs)):
                    if not pics:
                        continue
                    fid = r["test_ids"][0]
                    for j, (p, lab) in enumerate(zip(pics, ("hand-eye", "diff-render"))):
                        ax = fig.add_axes([0.03 + j * 0.485, 0.89 - (i + 1) * h, 0.47, h - 0.02])
                        ax.imshow(plt.imread(p))
                        ax.axis("off")
                        k = "he" if j == 0 else "dr"
                        ax.set_title(f"{r['name']}  {lab}: t {r[k + '_t'][0]:.2f} mm, rot {r[k + '_r'][0]:.2f}°, "
                                     f"IoU {r[k + '_iou'][0]:.3f}", fontsize=8)
                pdf.savefig(fig)
                plt.close(fig)
            shutil.rmtree(tmp, ignore_errors=True)

        d = pdf.infodict()
        d["Title"] = f"Markerless calibration cumulative evaluation ({N} runs)"
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
