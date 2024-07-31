# .
# ├── datasets
# │   └── UNIMIB2016
# │       ├── UNIMIB2016-annotations
# │       ├── images
# │       ├── labels
# │       └── split
# └── yolov5
#     └── labels_shower.py <--

# labels_shower.py

import os
import yaml
import numpy as np
from random import sample
from utils.general import xywhn2xyxy
from utils.plots import Annotator
from utils.general import cv2
from utils.dataloaders import LoadImages
from utils.plots import Colors

n = 5  # how many images you want to show

# file path set

# ../datasets/UNIMIB2016/labels/
labels_path = os.path.join(os.path.pardir, 'datasets', 'UNIMIB2016', 'labels')
# ../datasets/UNIMIB2016/images/
imgs_path = os.path.join(os.path.pardir, 'datasets', 'UNIMIB2016', 'images')
# data/UNIMIB2016.yaml
cls_path = os.path.join(os.getcwd(), 'data', 'UNIMIB2016.yaml')

# model data preparation
# you shouldn't change them
pt = True
stride = 2
imgsz = (640, 640)
datasets = os.listdir(labels_path)
line_thickness = 3  # bounding box thickness (pixels)
colors = Colors()  # create instance for 'from utils.plots import colors'
with open(cls_path, errors='ignore') as f:
    names = yaml.safe_load(f)['names']  # class names


def labels_shower():
    sources = sample(datasets, n)

    for source in sources:
        # Add bbox to image
        with open(os.path.join(labels_path, source)) as file:
            lines = file.readlines()
            dataset = LoadImages(os.path.join(imgs_path, source[:-4] + '.jpg'),
                                 img_size=imgsz, stride=stride, auto=pt)
            im0s = dataset.__iter__().__next__()[2]
            im0 = im0s.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            for line in lines:
                annot = line.split()
                c = int(annot[0])  # integer class
                label = names[c]
                xywhn = np.asarray([[float(i) for i in annot[1:]]])
                xyxy = xywhn2xyxy(xywhn, w=annotator.im.shape[1], h=annotator.im.shape[0])
                annotator.box_label(xyxy.tolist()[0], label, color=colors(c, True))

            im0 = annotator.result()

            cv2.imshow(str(source[:-4] + '.jpg'), im0)
            # press ESC to destroy cv2 windows
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    labels_shower()
