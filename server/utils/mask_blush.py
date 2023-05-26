#%%
import os
import cv2
import numpy as np

def mask_blush(img_path:os.PathLike):
    """
    White LED로 촬영한 사진에서 홍조가 있는 부분을\n
    붉기 정도에 따라 마스킹하여 강조한 이미지의 배열과 \n
    마스킹한 부위가 차지하는 비율을 반환합니다.\n
    ---
    ## Parameters:  
        img_path (str): 이미지의 경로  
      
    ## Returns:  
        output_array (ndarray): 원본 이미지의 배열과 마스킹의 배열을 합성한 배열\n 
        ratio (float): 전체 면적중 마스킹된 부분의 비율  
    """
    img_array = cv2.imread(img_path)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    height, width, _ = img_array.shape
    total_area = height * width

    lab_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l_array, a_array, b_array = lab_array[..., 0], lab_array[..., 1], lab_array[..., 2]

    masked_array = img_array.copy()
    ignore_mask = np.zeros_like(l_array, dtype=bool)
    blush_mask = np.zeros_like(l_array, dtype=np.uint8)

    ignore_mask[(l_array < 120) & (a_array < 140)] = True
    ignore_mask[b_array > 140] = True

    blush_mask[a_array > 140] = 1 
    blush_mask[a_array > 145] = 2 
    blush_mask[a_array > 150] = 3
    blush_mask[a_array > 155] = 4

    masked_array[..., 0][blush_mask==4] = 255
    masked_array[..., 1][blush_mask==4] = 0
    masked_array[..., 2][blush_mask==4] = 0

    masked_array[..., 0][blush_mask==3] = 255
    masked_array[..., 1][blush_mask==3] = 50
    masked_array[..., 2][blush_mask==3] = 50

    masked_array[..., 0][blush_mask==2] = 255
    masked_array[..., 1][blush_mask==2] = 100
    masked_array[..., 2][blush_mask==2] = 100

    masked_array[..., 0][blush_mask==1] = 255
    masked_array[..., 1][blush_mask==1] = 150
    masked_array[..., 2][blush_mask==1] = 150

    blush_score = blush_mask.sum()
    ratio = blush_score /total_area
    return masked_array, ratio

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    ROOT_PATH = "../dataset/skin/white"
    imgs_list = os.listdir(ROOT_PATH)
    for i in range(100):
        img_path = os.path.join(ROOT_PATH, imgs_list[i])
        img_array = cv2.imread(img_path)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        masked_array, ratio = mask_blush(img_path)
        print(ratio)
        print()
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.title('original')
        plt.imshow(img_array)
        plt.subplot(122)
        plt.title('masked')
        plt.imshow(masked_array)
        plt.show()
# %%
