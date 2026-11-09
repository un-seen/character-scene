"""Maya s T-shirt (MakeHuman female_casualsuit01, CC0): the chest logo painted out and the shirt turned
from blue to terracotta (only the shirt s blue: the orange trim and the jeans stay), 2026-11-02.
  ~/vllm-env/bin/python humans/tee.py   (original kept in humans/orig_tex/)"""
import os

import cv2
import numpy as np

D = os.path.expanduser("~/.config/blender/5.1/extensions/.user/user_default/mpfb/data/clothes/female_casualsuit01")
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orig_tex", "female_casualsuit01_diffuse.png")
im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
bgr = im[..., :3].copy()
H, W = bgr.shape[:2]
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
# the logo: what is not the shirt s blue inside its ring
ring = np.zeros((H, W), np.uint8); cv2.ellipse(ring, (int(0.795 * W), int(0.238 * H)), (int(0.080 * W), int(0.090 * H)), 0, 0, 360, 255, -1)
blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 90)
mask = ((ring > 0) & ~blue).astype(np.uint8) * 255
mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
bgr = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
# the shirt s panels: the top of the sheet and the two sleeves at the bottom right
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
shirt = np.zeros((H, W), bool); shirt[: int(0.43 * H)] = True; shirt[int(0.57 * H):, int(0.735 * W):] = True
blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 60) & shirt
h = hsv[..., 0].astype(int); h[blue] = (h[blue] - 103 + 8) % 180          # blue (~206 deg) to terracotta (~16 deg)
hsv[..., 0] = h.astype(np.uint8)
s = hsv[..., 1].astype(float); s[blue] *= 0.85; hsv[..., 1] = s.astype(np.uint8)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
out = np.dstack([bgr, im[..., 3]]) if im.shape[2] == 4 else bgr
cv2.imwrite(os.path.join(D, "female_casualsuit01_diffuse.png"), out)
cv2.imwrite(os.path.expanduser("~/character/humans/look/tee_tex.jpg"), cv2.resize(bgr, (800, 800)))
print("tee done", int(mask.sum() / 255), "logo px,", int(blue.sum()), "shirt px")
