"""Leo's white tee (MakeHuman male_casualsuit06, CC0): the MakeHuman logos on the chest, the back and
the sleeve painted out, 2026-11-09. Original kept in humans/orig_tex/.
  ~/vllm-env/bin/python humans/tee06.py"""
import os, shutil
import cv2
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
dst = os.path.expanduser("~/.config/blender/5.1/extensions/.user/user_default/mpfb/data/clothes/male_casualsuit06/male_casualsuit06_diffuse.png")
src = os.path.join(H, "orig_tex", "male_casualsuit06_diffuse.png")
if not os.path.exists(src): shutil.copy(dst, src)
im = cv2.imread(src)
h, w = im.shape[:2]
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
box = np.zeros((h, w), bool)
for x0, y0, x1, y1 in ((0.18, 0.10, 0.31, 0.17), (0.43, 0.11, 0.63, 0.22), (0.72, 0.14, 0.76, 0.18)):
    box[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
ink = (hsv[..., 1] > 60) | (hsv[..., 2] < 190)          # the orange figure, the grey letters and ring
mask = (box & ink).astype(np.uint8) * 255
mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
out = cv2.inpaint(im, mask, 9, cv2.INPAINT_TELEA)
cv2.imwrite(dst, out)
os.makedirs(os.path.join(H, "look"), exist_ok=True)
cv2.imwrite(os.path.join(H, "look", "tee06.jpg"), cv2.resize(out[: h // 2], (768, 384)))
print("tee06 done", round(float((mask > 0).mean()), 4))
