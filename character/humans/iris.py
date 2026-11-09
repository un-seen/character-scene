"""The brown eyes (MakeHuman eyes/materials/brown_eye.png, CC0) as a natural dark brown, 2026-11-02:
its iris is a red brown that reads as red on the page. Iris pixels (the saturated ones) turn to an
umber hue, less saturated and darker; the sclera and pupil stay. Original in humans/orig_tex/.
  ~/vllm-env/bin/python humans/iris.py"""
import os

import cv2
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
dst = os.path.expanduser("~/.config/blender/5.1/extensions/.user/user_default/mpfb/data/eyes/materials/brown_eye.png")
im = cv2.imread(os.path.join(H, "orig_tex", "brown_eye.png"), cv2.IMREAD_UNCHANGED)
hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV).astype(np.float32)
w = np.clip((hsv[..., 1] - 70) / 60, 0, 1) * ((hsv[..., 0] < 25) | (hsv[..., 0] > 160))   # iris: saturated red-brown (the sclera is pale)
hsv[..., 0] = hsv[..., 0] * (1 - w) + 14 * w               # OpenCV hue 14 = 28 deg: umber
hsv[..., 1] *= 1 - 0.35 * w
hsv[..., 2] *= 1 - 0.35 * w
out = im.copy(); out[..., :3] = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
cv2.imwrite(dst, out)
cv2.imwrite(os.path.join(H, "look", "iris_new.png"), cv2.resize(out[..., :3], (512, 512)))
print("iris done", round(float((w > 0.5).mean()), 3))
