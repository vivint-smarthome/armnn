#!/usr/bin/env python3

import os
import sys

def add_bbox(score, x0, y0, x1, y1):
   return "-stroke green -fill none -draw \"rectangle {},{} {},{}\" ".format(int(x0+.5), int(y0+.5), int(x1+.5), int(y1+.5))

def main(image_file, data_file, output_image_file):
    sz = os.stat(image_file).st_size
    model_w = 0
    model_h = 0
    print(sz)
    if sz == 512*256*3:
        model_w = 512
        model_h = 256
    elif sz == 320*320*3:
        model_w = 320
        model_h = 320
    else:
        return 1

    # Read bboxes
    bboxes = []
    scores = []
    with open(data_file) as f:
        content = f.readlines()
        for l in content:
            if l.startswith("BBox"):
                bbox = float(l[l.find(": ")+2:].rstrip())
                bboxes.append(bbox)
            if l.startswith("Score"):
                score = float(l[l.find(": ")+2:].rstrip())
                scores.append(score)

    cmd = "gm convert -size {}x{} -depth 8 {} ".format(model_w, model_h, image_file)
    i = 0
    while i < len(bboxes):
        ymin = bboxes[i] 
        xmin = bboxes[i+1] 
        ymax = bboxes[i+2] 
        xmax = bboxes[i+3] 

        print(ymin, xmin, ymax, xmax)

        cmd += add_bbox(scores[i//4], xmin*model_w, ymin*model_h, xmax*model_w, ymax*model_h)
        i += 4

    cmd += " " + output_image_file

    print(cmd)

    os.system(cmd)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
