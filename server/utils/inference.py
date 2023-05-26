import torch
import torchvision
import cv2
from utils import draw_bbox
import os

def inference(img_path, save_path, model, device, score_tres):
    transforms = torchvision.transforms.ToTensor()

    img_array = cv2.imread(img_path)
    img_tensor = transforms(img_array)
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        pred = model([img_tensor])
    boxes = pred[0]['boxes']
    labels = pred[0]['labels']
    scores = pred[0]['scores']

    boxes = boxes[scores > score_tres]
    labels = labels[scores > score_tres]

    boxes_list = boxes.tolist()
    labels_list = labels.tolist()

    labels_list = ['sebum' for _ in labels_list]

    out_img = draw_bbox(img_array= img_array,
                        label_list=labels_list,
                        bboxes_list=boxes_list)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, out_img)