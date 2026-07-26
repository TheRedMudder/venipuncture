#!/usr/bin/env python3
"""
Hand Dark-Area Stream Analyzer
==============================
Real-time version of the hand dark-area pipeline, built for video input
(~15 fps or faster). Feed it a video file, a camera, or push frames to it
from your own loop.

What it computes per frame (same logic as the single-image version):
  1. Hand segmentation (bright hand vs dark background)
  2. All dark areas (veins/shadows) inside the hand, ranked by size
  3. Largest all-dark spot (max inscribed circle) inside region #1

Key optimizations vs. the single-image script:
  * All heavy work runs on a DOWNSCALED copy (default 640 px wide);
    results are mapped back to full resolution. ~30-100x less pixels.
  * The hand mask is re-computed only every N frames (default 5) and
    reused in between -- the hand moves slowly relative to the veins.
  * Distance transform runs only on region #1's bounding-box crop,
    not the whole frame.
  * Kernels are pre-built once; no per-frame allocations of structuring
    elements; morphology sizes scale with the working resolution.
  * Optional exponential smoothing of the dark mask across frames to
    suppress single-frame flicker.
  * Threaded frame grabber that always hands you the LATEST frame, so
    processing lag never builds an ever-growing backlog (old frames are
    dropped, which is what you want for live streams).

Usage:
  # video file, annotated preview window + stats:
  python hand_stream_analyzer.py input.mp4 --display

  # webcam / capture device:
  python hand_stream_analyzer.py 0 --display

  # save annotated output video and per-frame JSONL report:
  python hand_stream_analyzer.py input.mp4 --save out.mp4 --report report.jsonl

  # library use:
  from hand_stream_analyzer import HandDarkAreaStreamAnalyzer
  an = HandDarkAreaStreamAnalyzer()
  for frame in my_frames:                    # BGR or grayscale np.ndarray
      result = an.process(frame)             # dict of measurements
      overlay = an.annotate(frame, result)   # optional visualization

Requires: opencv-python, numpy
"""

import argparse
import json
import sys
import threading
import time

import cv2
import numpy as np


# ======================================================================
# Core analyzer
# ======================================================================
class HandDarkAreaStreamAnalyzer:
    def __init__(
        self,
        proc_width: int = 640,        # working resolution (px). Lower = faster.
        hand_interval: int = 5,       # re-segment the hand every N frames
        block_frac: float = 0.028,    # adaptive block size as fraction of width
        c: int = 6,                   # adaptive threshold sensitivity
        min_area_frac: float = 6e-5,  # min region size as fraction of frame area
        mask_smooth: float = 0.5,     # 0 = off; else EMA weight of current frame
        top_k: int = 10,              # how many regions to report
    ):
        self.proc_width = proc_width
        self.hand_interval = max(1, hand_interval)
        self.block_frac = block_frac
        self.c = c
        self.min_area_frac = min_area_frac
        self.mask_smooth = mask_smooth
        self.top_k = top_k

        # pre-built kernels (sized for the working resolution)
        s = max(1, proc_width // 640)
        self._k_close_hand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5 * s, 5 * s))
        self._k_erode_hand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7 * s, 7 * s))
        self._k_open_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_close_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # temporal state
        self._frame_idx = 0
        self._hand_inner = None       # cached eroded hand mask (working res)
        self._hand_area_full = 0
        self._mask_ema = None         # float32 EMA of the dark mask
        self._scale = None            # working-res -> full-res factor

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> dict:
        """Analyze one frame. Accepts BGR or grayscale. Returns a dict with
        all measurements in FULL-resolution coordinates."""
        t0 = time.perf_counter()

        gray_full = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray_full.shape

        # ---- downscale once, everything below runs at working resolution
        scale = self.proc_width / W
        h_w = int(round(H * scale))
        gray = cv2.resize(gray_full, (self.proc_width, h_w), interpolation=cv2.INTER_AREA)
        self._scale = 1.0 / scale
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # ---- 1) hand mask: recompute only every `hand_interval` frames
        if self._hand_inner is None or self._frame_idx % self.hand_interval == 0:
            self._hand_inner, hand_area_w = self._segment_hand(blur)
            self._hand_area_full = int(hand_area_w * self._scale ** 2)
        hand_inner = self._hand_inner

        # ---- 2) dark areas inside the hand (adaptive threshold)
        block = int(self.block_frac * self.proc_width) | 1  # force odd
        block = max(11, block)
        adap = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, self.c,
        )
        dark = cv2.bitwise_and(adap, hand_inner)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, self._k_open_dark)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, self._k_close_dark)

        # ---- temporal smoothing (EMA) to kill single-frame flicker
        if self.mask_smooth > 0:
            f = dark.astype(np.float32)
            if self._mask_ema is None or self._mask_ema.shape != f.shape:
                self._mask_ema = f
            else:
                a = self.mask_smooth
                cv2.addWeighted(f, a, self._mask_ema, 1 - a, 0, dst=self._mask_ema)
            dark = (self._mask_ema > 127).astype(np.uint8) * 255

        # ---- connected components + ranking
        min_area_w = max(8, int(self.min_area_frac * self.proc_width * h_w))
        n, lbl, stats, cents = cv2.connectedComponentsWithStats(dark)
        regions = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_w:
                continue
            regions.append((i, area, stats[i], cents[i]))
        regions.sort(key=lambda r: -r[1])
        regions = regions[: self.top_k]

        S = self._scale
        out_regions = []
        for rank, (i, area, st, ce) in enumerate(regions, 1):
            x, y, w, h = (int(v) for v in st[:4])
            out_regions.append({
                "rank": rank,
                "area_px": int(area * S * S),
                "bbox": [int(x * S), int(y * S), int(w * S), int(h * S)],
                "centroid": [int(ce[0] * S), int(ce[1] * S)],
            })

        # ---- 3) largest all-dark circle inside region #1 (bbox crop only)
        circle = None
        if regions:
            i, _, st, _ = regions[0]
            x, y, w, h = (int(v) for v in st[:4])
            crop = np.where(lbl[y:y + h, x:x + w] == i, 255, 0).astype(np.uint8)
            dist = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
            _, radius, _, loc = cv2.minMaxLoc(dist)
            circle = {
                "center": [int((x + loc[0]) * S), int((y + loc[1]) * S)],
                "radius_px": round(radius * S, 1),
                "area_px": int(np.pi * (radius * S) ** 2),
            }

        dark_area_full = int(np.sum(dark == 255) * S * S)
        self._frame_idx += 1
        return {
            "frame": self._frame_idx - 1,
            "hand_area_px": self._hand_area_full,
            "dark_area_px": dark_area_full,
            "dark_pct_of_hand": round(100 * dark_area_full / max(self._hand_area_full, 1), 2),
            "regions": out_regions,
            "region1_largest_all_dark_circle": circle,
            "proc_ms": round((time.perf_counter() - t0) * 1000, 2),
            "_mask_small": dark,       # internal, for annotate()
        }

    # ------------------------------------------------------------------
    def _segment_hand(self, blur_small: np.ndarray):
        _, mask = cv2.threshold(blur_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = np.where(lbl == biggest, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close_hand)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, cnts, -1, 255, -1)
        inner = cv2.erode(filled, self._k_erode_hand)
        return inner, int(np.sum(filled == 255))

    # ------------------------------------------------------------------
    def annotate(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw the results on a copy of the (full-res) frame."""
        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        mask_small = result.get("_mask_small")
        if mask_small is not None:
            mask = cv2.resize(mask_small, (vis.shape[1], vis.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            red = vis.copy()
            red[mask == 255] = (0, 0, 255)
            vis = cv2.addWeighted(vis, 0.6, red, 0.4, 0)
        t = max(1, vis.shape[1] // 800)
        for r in result["regions"]:
            x, y, w, h = r["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), t)
            cv2.putText(vis, f"#{r['rank']}", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 255, 0), t)
        c = result["region1_largest_all_dark_circle"]
        if c:
            cv2.circle(vis, tuple(c["center"]), int(c["radius_px"]), (255, 255, 0), 2 * t)
        cv2.putText(vis, f"{result['proc_ms']:.1f} ms  dark={result['dark_pct_of_hand']}%",
                    (10, 25 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (255, 255, 255), t)
        return vis


# ======================================================================
# Threaded latest-frame grabber (drops stale frames -> no lag buildup)
# ======================================================================
class LatestFrameGrabber:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                self.stopped = True
                break
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            f, self.frame = self.frame, None
        return f

    def release(self):
        self.stopped = True
        self.t.join(timeout=1)
        self.cap.release()


# ======================================================================
# CLI runner
# ======================================================================
def main():
    p = argparse.ArgumentParser(description="Stream hand dark-area analysis on video.")
    p.add_argument("source", help="Video file path, or camera index (e.g. 0).")
    p.add_argument("--proc-width", type=int, default=640,
                   help="Working resolution width (default 640). Lower = faster.")
    p.add_argument("--hand-interval", type=int, default=5,
                   help="Re-segment the hand every N frames (default 5).")
    p.add_argument("--fps", type=float, default=15,
                   help="Target processing rate for file input (default 15).")
    p.add_argument("--live", action="store_true",
                   help="Treat source as live: always process the newest frame, drop stale ones.")
    p.add_argument("--display", action="store_true", help="Show annotated preview window.")
    p.add_argument("--save", default=None, help="Save annotated video to this path.")
    p.add_argument("--report", default=None, help="Write per-frame JSON lines to this file.")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all).")
    args = p.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    analyzer = HandDarkAreaStreamAnalyzer(
        proc_width=args.proc_width, hand_interval=args.hand_interval)

    writer = None
    report_f = open(args.report, "w") if args.report else None
    frame_budget = 1.0 / args.fps if args.fps > 0 else 0

    if args.live:
        grab = LatestFrameGrabber(source)
        get = grab.read
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit(f"Cannot open: {args.source}")
        get = lambda: cap.read()[1]

    n_done, t_start = 0, time.perf_counter()
    try:
        while True:
            t_frame = time.perf_counter()
            frame = get()
            if frame is None:
                if args.live and not grab.stopped:
                    time.sleep(0.002)      # waiting for a fresh live frame
                    continue
                break

            result = analyzer.process(frame)

            if args.display or args.save:
                vis = analyzer.annotate(frame, result)
                if args.save:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(args.save, fourcc, args.fps,
                                                 (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if args.display:
                    cv2.imshow("hand dark areas", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if report_f:
                clean = {k: v for k, v in result.items() if not k.startswith("_")}
                report_f.write(json.dumps(clean) + "\n")

            n_done += 1
            if args.max_frames and n_done >= args.max_frames:
                break

            # pace file playback to the target fps (live mode self-paces)
            if not args.live and frame_budget:
                spare = frame_budget - (time.perf_counter() - t_frame)
                if spare > 0:
                    time.sleep(spare)
    finally:
        elapsed = time.perf_counter() - t_start
        if args.live:
            grab.release()
        else:
            cap.release()
        if writer:
            writer.release()
        if report_f:
            report_f.close()
        cv2.destroyAllWindows()
        if n_done:
            print(f"Processed {n_done} frames in {elapsed:.1f}s "
                  f"-> {n_done / elapsed:.1f} fps effective")


if __name__ == "__main__":
    main()
