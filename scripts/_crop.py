import sys
from PIL import Image
im = Image.open(sys.argv[1]); w, h = im.size; x0, y0, x1, y1 = [float(v) for v in sys.argv[3:7]]
im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).save(sys.argv[2])
