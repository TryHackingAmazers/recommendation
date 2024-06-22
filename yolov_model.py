# import os

import numpy as np

from ultralytics import YOLO
# import cv2



def train():
    dataset_config_path = "../datasets/furniture-detection-20/data.yaml"
    model = YOLO("checkpoint/yolov8n.pt")
    model.train(data=dataset_config_path, epochs=10)

def load():
    model = YOLO("/home/rohan/hackonama/recommendation/checkpoint/best.pt")
    # model = YOLO("checkpoint/yolov8l.pt")
    return model


def process_image(inp_image):
    model = load()
    result = model(inp_image,verbose=False)
    objects = result[0].boxes.data.cpu().numpy()
    labels_map = result[0].names
    # image = cv2.imread(image_path)

    score = np.zeros(len(labels_map))
    # Iterate over each object
    for i in range(len(objects)):
        # Get the bounding box coordinates
        x1, y1, x2, y2, p, label = objects[i]
        score[int(label)] += 1
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Draw the bounding box
        # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the label
        # cv2.putText(image, labels_map[label], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Save the image
    # cv2.imwrite('output1.jpg', image)
    return score

