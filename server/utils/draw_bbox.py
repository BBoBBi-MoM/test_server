import cv2
import numpy as np

def draw_bbox(img_array:np.ndarray,
              label_list:list,
              bboxes_list:list)-> np.ndarray:
    
    assert len(label_list) == len(bboxes_list)
    
    box_on_img = img_array.copy()
    num_bboxes = len(bboxes_list)
    for i in range(num_bboxes):
        box = bboxes_list[i]
        label = label_list[i]
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        cv2.rectangle(img=box_on_img, 
                      pt1=(xmin, ymin), pt2=(xmax, ymax), 
                      color=(0, 0, 255), thickness=1)
        
        cv2.putText(img=box_on_img, text=label, org=(xmin, ymin-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=(0, 0, 255), thickness=2)
    return box_on_img