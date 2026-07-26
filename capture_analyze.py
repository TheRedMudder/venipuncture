#!/usr/bin/env python3
"""
capture_analyze.py
==================
Basler capture piped straight into HandDarkAreaStreamAnalyzer.

Per frame: grab -> CLAHE enhance -> analyzer.process() -> annotated panel.
The analyzer runs on the ENHANCED image by default: CLAHE deepens local
vein contrast, and the analyzer's adaptive threshold keys off local
contrast, so the enhanced input gives it more to grab. Press E to feed
it the RAW frame instead and compare.

Panels:  RAW | ENHANCED | ANALYZED  (+ VESSELS when Frangi is on)

Keys:
  S            save raw/enhanced/annotated PNGs + analysis JSON
  A            toggle analysis on/off
  E            analyzer input: enhanced <-> raw
  J            toggle continuous JSONL logging of per-frame results
  F            toggle Frangi vessels panel
  Q            quit
  Up/Down      exposure +/- 1000 us
  Left/Right   CLAHE clip +/- 0.5

Files land in captures/YYYY-MM-DD/ next to your existing saves.
Requires hand_stream_analyzer.py in the same folder.
"""

import json
import os
import time

import cv2
import numpy as np
from pypylon import pylon

from hand_stream_analyzer import HandDarkAreaStreamAnalyzer

# --- CAPTURE RECIPE ---------------------------------------------------------
# Lock these in as you calibrate. These are the session-to-session constants.
EXPOSURE_US = 6000.0     # tune with Up/Down, then write the winner here
GAIN_DB = 0.0            # keep 0; gain adds noise, the ring light adds signal
CLAHE_CLIP = 3.0         # tune with Left/Right, then write the winner here
CLAHE_TILES = (8, 8)
WORKING_DISTANCE_CM = 25  # doc only - where the arm sits; keep it consistent
DIMMER_NOTE = "ring dimmer at ___"  # write the physical knob position here

DISPLAY_WIDTH = 520      # per panel on screen; saves are full resolution
FRANGI_WIDTH = 1200      # Frangi runs downscaled (full-res is very slow)

# --- ANALYZER RECIPE --------------------------------------------------------
PROC_WIDTH = 640         # analyzer working resolution; lower = faster
HAND_INTERVAL = 5        # re-segment the hand every N frames


def make_analyzer():
    """Fresh analyzer = fresh temporal state (hand cache + mask EMA).
    Rebuilt whenever the input source changes so stale masks don't bleed."""
    return HandDarkAreaStreamAnalyzer(
        proc_width=PROC_WIDTH,
        hand_interval=HAND_INTERVAL,
    )


# --- Camera -----------------------------------------------------------------
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ExposureAuto.SetValue("Off")
camera.GainAuto.SetValue("Off")
camera.ExposureTime.SetValue(EXPOSURE_US)
camera.Gain.SetValue(GAIN_DB)
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

exposure = EXPOSURE_US
clip = CLAHE_CLIP
show_frangi = False

analyzer = make_analyzer()
analyze = True           # A toggles
analyze_enhanced = True  # E toggles: True = analyzer sees CLAHE output
log_f = None             # J toggles a per-frame JSONL log

try:
    from skimage.filters import frangi
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False
    print("scikit-image not installed - F key disabled "
          "(pip install scikit-image)")


def enhance(img, clip_limit):
    """Denoise then CLAHE. Contrast lives in the midtones."""
    smoothed = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=CLAHE_TILES)
    return clahe.apply(smoothed)


def vessels(img):
    """Frangi vesselness: responds to dark tube-like structures.
    Runs on a downscaled copy for speed; returns uint8 vessel map."""
    scale = FRANGI_WIDTH / img.shape[1]
    small = cv2.resize(img, (FRANGI_WIDTH, int(img.shape[0] * scale)))
    v = frangi(small.astype(np.float32) / 255.0, black_ridges=True)
    v = (255 * (v / (v.max() + 1e-9))).astype(np.uint8)
    return v


def sharpness(img):
    """Variance of Laplacian on the center crop. Higher = sharper.
    Relative number - compare against your own in-focus baseline."""
    h, w = img.shape
    crop = img[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    return cv2.Laplacian(crop, cv2.CV_64F).var()


def panel(img, label, width):
    """Resize + label. Always returns BGR so panels concat with the
    color annotated view."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = width / img.shape[1]
    p = cv2.resize(img, (width, int(img.shape[0] * scale)))
    cv2.putText(p, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)
    return p


def clean_result(result, exposure_us, clahe_clip, on_enhanced):
    """Strip internal keys and stamp the capture settings so every saved
    measurement carries its calibration context."""
    out = {k: v for k, v in result.items() if not k.startswith("_")}
    out["exposure_us"] = int(exposure_us)
    out["clahe_clip"] = round(clahe_clip, 1)
    out["analyzer_input"] = "enhanced" if on_enhanced else "raw"
    return out


print(__doc__)

while camera.IsGrabbing():
    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
    if not grab.GrabSucceeded():
        grab.Release()
        continue
    frame = grab.Array
    grab.Release()

    enhanced = enhance(frame, clip)

    # ---- pipe into the analyzer -------------------------------------------
    result = None
    annotated = None
    if analyze:
        target = enhanced if analyze_enhanced else frame
        result = analyzer.process(target)
        annotated = analyzer.annotate(target, result)

    if log_f is not None and result is not None:
        entry = clean_result(result, exposure, clip, analyze_enhanced)
        entry["ts"] = round(time.time(), 3)
        log_f.write(json.dumps(entry) + "\n")

    # ---- panels -----------------------------------------------------------
    panels = [panel(frame, "RAW", DISPLAY_WIDTH),
              panel(enhanced, "ENHANCED", DISPLAY_WIDTH)]
    if annotated is not None:
        src = "ENH" if analyze_enhanced else "RAW"
        panels.append(panel(annotated, f"ANALYZED ({src})", DISPLAY_WIDTH))

    frangi_map = None
    if show_frangi and HAVE_SKIMAGE:
        frangi_map = vessels(enhanced)
        panels.append(panel(frangi_map, "VESSELS", DISPLAY_WIDTH))

    # rounding in the resize chains can leave panels 1 px apart in height;
    # normalize so hconcat never trips
    h0 = panels[0].shape[0]
    panels = [p if p.shape[0] == h0 else cv2.resize(p, (p.shape[1], h0))
              for p in panels]
    combo = cv2.hconcat(panels)

    # ---- status bar -------------------------------------------------------
    status = (f"exp {int(exposure)}us | clip {clip:.1f} | "
              f"sharp {sharpness(frame):.0f}")
    if result is not None:
        status += (f" | dark {result['dark_pct_of_hand']}% | "
                   f"regions {len(result['regions'])} | "
                   f"{result['proc_ms']:.0f} ms")
    if log_f is not None:
        status += " | LOGGING"
    status += " | S=save A=ana E=input J=log F=frangi Q=quit arrows=tune"
    bar = np.zeros((34, combo.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.imshow("Viewer", cv2.vconcat([combo, bar]))

    # ---- keys -------------------------------------------------------------
    key = cv2.waitKeyEx(1)
    if key == -1:
        continue
    k = key & 0xFF

    if k == ord("q"):
        break
    elif k == ord("a"):
        analyze = not analyze
        if analyze:
            analyzer = make_analyzer()
        print(f"analysis {'on' if analyze else 'off'}")
    elif k == ord("e"):
        analyze_enhanced = not analyze_enhanced
        analyzer = make_analyzer()
        print(f"analyzer input: {'enhanced' if analyze_enhanced else 'raw'}")
    elif k == ord("j"):
        if log_f is not None:
            log_f.close()
            log_f = None
            print("jsonl logging stopped")
        else:
            day_dir = os.path.join("captures", time.strftime("%Y-%m-%d"))
            os.makedirs(day_dir, exist_ok=True)
            path = os.path.join(day_dir,
                                f"report_{time.strftime('%H%M%S')}.jsonl")
            log_f = open(path, "w")
            print(f"jsonl logging -> {path}")
    elif k == ord("f") and HAVE_SKIMAGE:
        show_frangi = not show_frangi
    elif k == ord("s"):
        day_dir = os.path.join("captures", time.strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        tag = f"exp{int(exposure)}_clip{clip:.1f}"
        cv2.imwrite(f"{day_dir}/raw_{stamp}_{tag}.png", frame)
        cv2.imwrite(f"{day_dir}/enh_{stamp}_{tag}.png", enhanced)
        if frangi_map is not None:
            cv2.imwrite(f"{day_dir}/ves_{stamp}_{tag}.png", frangi_map)
        if annotated is not None:
            cv2.imwrite(f"{day_dir}/ana_{stamp}_{tag}.png", annotated)
        if result is not None:
            with open(f"{day_dir}/ana_{stamp}_{tag}.json", "w") as jf:
                json.dump(clean_result(result, exposure, clip,
                                       analyze_enhanced), jf, indent=2)
        print(f"saved -> {day_dir}/*_{stamp}_{tag}.*")
    # Arrow keys arrive as extended codes; cover common macOS/Linux values
    elif key in (63232, 2490368, 82):      # Up
        exposure = min(exposure + 1000, 100000)
        camera.ExposureTime.SetValue(exposure)
    elif key in (63233, 2621440, 84):      # Down
        exposure = max(exposure - 1000, 500)
        camera.ExposureTime.SetValue(exposure)
    elif key in (63235, 2555904, 83):      # Right
        clip = min(clip + 0.5, 8.0)
    elif key in (63234, 2424832, 81):      # Left
        clip = max(clip - 0.5, 0.5)

if log_f is not None:
    log_f.close()
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
