import math

import cv2
import numpy as np

from bbox.converters import corner_form_to_center_form
from dataset.augmentation.transforms import PredictionTransform
from model.ssd.mobilenet import mobileV1_ssd_config
from model.ssd.mobilenet.mobileV1_ssd import create_mobilenetv1_ssd
from model.ssd.mobilenet.mobileV1_ssd_config import CONFIG
from model.ssd.predictor import Predictor
from model.ssd.ssd import SSDTest
from train import DEVICE


def demo():
    model_path = 'checkpoint/mobilev1-ssd-Epoch-90-Loss-1.2602647761503856.pth'
    image_path = '/home/truewarg/data/fake-test-3/VOC2007-fake-3/JPEGImages/image_50.png'

    class_labels = ('BACKGROUND', 'red', 'green', 'blue')

    net = create_mobilenetv1_ssd(len(class_labels))
    net.load(model_path)
    net = SSDTest(
        ssd=net,
        config=CONFIG,
        priors=mobileV1_ssd_config.priors,
    )
    predictor = Predictor(
        net=net,
        transform=PredictionTransform(CONFIG.image_size, CONFIG.image_mean, CONFIG.image_std),
        iou_threshold=CONFIG.iou_threshold,
        candidate_size=200,
        device=DEVICE,
    )

    orig_image = cv2.imread(image_path)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        box = corner_form_to_center_form(box)
        center_x, center_y, width, height, angle = box
        box_points = cv2.boxPoints((
            (center_x, center_y),
            (width, height),
            (angle * 180) // math.pi,
        ))

        box_points = np.int32(box_points)

        thickness = 1
        contour_idx = 0

        cv2.drawContours(orig_image, [box_points], contour_idx, (255, 255, 0), thickness)

        label = f"{class_labels[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    path = "run_ssd_example_output.jpg"
    cv2.imwrite(path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {path}")


if __name__ == '__main__':
    demo()
