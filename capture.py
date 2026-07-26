import os
import time

import cv2
import numpy as np
from pypylon import pylon

# --- CAPTURE RECIPE ---------------------------------------------------------
# Lock these in as you calibrate. These are the session-to-session constants.
EXPOSURE_US = 6000.0     # tune with Up/Down, then write the winner here
GAIN_DB = 0.0            # keep 0; gain adds noise, the ring light adds signal
CLAHE_CLIP = 3.0         # tune with Left/Right, then write the winner here
CLAHE_TILES = (8, 8)
WORKING_DISTANCE_CM = 25  # doc only — where the arm sits; keep it consistent
DIMMER_NOTE = "ring dimmer at ___"  # write the physical knob position here

DISPLAY_WIDTH = 520      # per panel on screen; saves are full resolution
FRANGI_WIDTH = 1200      # Frangi runs downscaled (full-res is very slow)

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

try:
    from skimage.filters import frangi
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False
    print("scikit-image not installed — F key disabled "
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
    # black_ridges=True
    v = frangi(small.astype(np.float32) / 255.0, black_ridges=True)
    v = (255 * (v / (v.max() + 1e-9))).astype(np.uint8)
    return v


def sharpness(img):
    """Variance of Laplacian on the center crop. Higher = sharper.
    Relative number — compare against your own in-focus baseline."""
    h, w = img.shape
    crop = img[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    return cv2.Laplacian(crop, cv2.CV_64F).var()


def panel(img, label, width):
    scale = width / img.shape[1]
    p = cv2.resize(img, (width, int(img.shape[0] * scale)))
    cv2.putText(p, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return p


print(__doc__)

while camera.IsGrabbing():
    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
    if not grab.GrabSucceeded():
        grab.Release()
        continue
    frame = grab.Array
    grab.Release()

    enhanced = enhance(frame, clip)

    panels = [panel(frame, "RAW", DISPLAY_WIDTH),
              panel(enhanced, "ENHANCED", DISPLAY_WIDTH)]

    frangi_map = None
    if show_frangi and HAVE_SKIMAGE:
        frangi_map = vessels(enhanced)
        panels.append(panel(frangi_map, "VESSELS", DISPLAY_WIDTH))

    combo = cv2.hconcat(panels)

    # Status bar: current settings + sharpness
    status = (f"exp {int(exposure)}us | clip {clip:.1f} | "
              f"sharp {sharpness(frame):.0f} | "
              f"S=save F=frangi Q=quit arrows=tune")
    bar = np.zeros((34, combo.shape[1]), dtype=np.uint8)
    cv2.putText(bar, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1)
    cv2.imshow("Viewer", cv2.vconcat([combo, bar]))

    key = cv2.waitKeyEx(1)
    if key == -1:
        continue
    k = key & 0xFF

    if k == ord("q"):
        break
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
        print(f"saved -> {day_dir}/*_{stamp}_{tag}.png")
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

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
